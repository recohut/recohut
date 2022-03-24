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

<!-- #region id="rENGvL1WaXBW" -->
# Session-based E-commerce Product Recommender
> We will build one of the simplest and powerful session-based recommender engine on a real-world data. The data contains [Trendyol's](https://www.trendyol.com/) session-level activities and product metadata information. 

- toc: false
- badges: true
- comments: true
- categories: [Session, Sequence, Retail, ECommerce]
- author: "<a href='https://github.com/CeyhanTurnali/ProductRecommendation'>CeyhanTurnalı</a>"
- image:
<!-- #endregion -->

```python id="E8gbN55l4FTu"
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from sklearn.metrics.pairwise import cosine_similarity
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="hmXEfQSe4L2n" outputId="8c079082-ae50-4e26-a55b-c9c96186b447"
meta = pd.read_parquet('https://github.com/recohut/reco-data/raw/trendyol/trendyol/v1/meta.parquet.gzip')
meta.head(5)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="byE20ov25ZA4" outputId="60659814-0241-4c10-f07d-4dbef3a3414e"
events = pd.read_parquet('https://github.com/recohut/reco-data/raw/trendyol/trendyol/v1/events.parquet.gzip')
events.head(5)
```

<!-- #region id="49D860wA5f8x" -->
There are two dataset which are contains prouducts and session details. I used productid as a primary key and merge two csv files.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="KlTZBdSA5d9a" outputId="604d91db-2a4b-4ce7-e2c9-e6fd70687e34"
data = meta.merge(events, on="productid")
data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="H1Bu0u2c5qBx" outputId="d49b3461-7076-499f-da1f-507494c125be"
data.info()
```

<!-- #region id="xSoB6tXeCrpA" -->
Identify and drop null in ids columns
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aDdixI2TCaIS" outputId="86efef8d-6c7d-4199-a707-0feff1b24a20"
data.isna().sum()
```

```python id="8-v_wyolCh1H"
data = data.dropna(subset=['sessionid','productid'])
```

```python colab={"base_uri": "https://localhost:8080/"} id="2o9ID1-RCqGQ" outputId="42d092bd-8275-4856-85a4-bd6a691889ec"
data.isna().sum()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="uRIy0SYJ6DEQ" outputId="c195f803-6839-4cbb-b439-92f021d36ca2"
data.describe(include=['O']).T
```

<!-- #region id="f6J5JwNY6Qh4" -->
Cart is a category but we can use it as a quantity. Every cart process is one buying and we can use it as a quantity to answer how many products did the customers buy.
<!-- #endregion -->

```python id="dRelTI5C6Hgn"
data['event'] = data['event'].replace(['cart'],'1')
data['event'] = data['event'].astype(float)
```

```python id="YGWRjWs-ZAfA"
data_full = data.copy()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="TqWupVSSBaB3" outputId="81c2f4da-38b6-408b-8f88-59a2d876e838"
data = data[['sessionid','productid','event']]
data.head()
```

<!-- #region id="E-Txbxuu6xSR" -->
Next, we will create a session-item matrix. In this matrix, each row represents a session, each column represents each product or item and the value in each cell indicates whether the customer has purchased the given product in that particular session.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ZFhB114kBTpA" outputId="bd8cde73-54e3-40b1-a98a-89fa1886d495"
session_c = CategoricalDtype(sorted(data.sessionid.unique()), ordered=True)
product_c = CategoricalDtype(sorted(data.productid.unique()), ordered=True)

row = data.sessionid.astype(session_c).cat.codes
col = data.productid.astype(product_c).cat.codes

session_item_matrix = csr_matrix((data["event"], (row, col)), shape=(session_c.categories.size, product_c.categories.size))
session_item_matrix
```

```python colab={"base_uri": "https://localhost:8080/"} id="ggNd2z4yELPb" outputId="7bd380af-976e-45df-dbbd-50f9ba18ab27"
session_item_matrix[:10,:10].todense()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="YaeYVzuiGRUz" outputId="36827e8b-2332-4bc6-e7f8-f1ff08829633"
session_c.categories[10]
```

<!-- #region id="zbDv4ey5AnGq" -->
## User-User Similarity
<!-- #endregion -->

<!-- #region id="icXDjBbb7QaG" -->
We compute the cosine similarity from the session item matrix to determine similarity between user's purchase behaviour.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="slHS1piKF41-" outputId="14a7fa53-e4cb-44cd-ceba-1b0d6e6338d2"
user_user_sim_matrix = cosine_similarity(session_item_matrix, dense_output=False)
user_user_sim_matrix
```

```python id="iaMqp_z3HWzo"
def getname(id=0, ntype='session', mode='lookup'):
  if mode=='random':
    if ntype=='session':
      id = np.random.randint(0,len(session_c.categories))
      return session_c.categories[id], id
    else:
      id = np.random.randint(0,len(product_c.categories))
      return product_c.categories[id], id
  else:
    if ntype=='session':
      return session_c.categories[id]
    else:
      return product_c.categories[id]
   

def print_topk(matrix, id, k=10, ntype='session'):
  frame = pd.DataFrame(matrix[id].todense()).T.sort_values(by=0, ascending=False).head(k)
  frame = frame.reset_index()
  frame.columns = ['id','similarity']
  frame[f'{ntype}_id'] = frame['id'].apply(lambda x: getname(x, ntype))
  return frame
```

```python colab={"base_uri": "https://localhost:8080/"} id="e-J-UrhYHTez" outputId="e531b1a7-0001-4ecb-dfc2-5531e7b7b949"
random_session, id = getname(ntype='session', mode='random')
print("Let's try it for a random session {}".format(random_session))
```

<!-- #region id="yaulgRot_bpQ" -->
What are the similar sessions?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="ay_k2TTHQnWK" outputId="424b8ecc-34da-4759-9f3a-eff1d095054a"
similar_sessions = print_topk(user_user_sim_matrix, id=id, k=10, ntype='session')
similar_sessions
```

```python colab={"base_uri": "https://localhost:8080/"} id="G9aU9OdIQlx6" outputId="d40e83b7-809f-473e-fa44-2de938114eb1"
print("Random Session ID: {}\nTop-similar Session ID: {}".\
      format(random_session, similar_sessions.iloc[1].session_id))
```

<!-- #region id="U52b-5jWSev_" -->
For reference, we take a random session id as A and top-most similar session id as B. Therefore, by identifying the items purchased by Customer A and Customer B and the Remaining Items of Customer A relative to Customer B, we can safely assume that there is high similarity between customers, as there is high similarity between customers. The rest of the products purchased by customer A are also likely to be purchased by customer B. Therefore, we recommend the remaining products to Customer
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="N6dqOt06_xTZ" outputId="50fa73c2-d8c5-495e-9de4-4813ecd0d036"
items_bought_by_customerA = [getname(x, ntype='product') for x in np.argwhere(session_item_matrix[id]>0)[:,1]]
print("Items Bought by Customer A:")
items_bought_by_customerA
```

```python colab={"base_uri": "https://localhost:8080/"} id="79MezSooACUP" outputId="89206cdc-fc19-48f7-f06c-a533dcfc7883"
items_bought_by_customerB = [getname(x, ntype='product') for x in np.argwhere(session_item_matrix[similar_sessions.iloc[1].id]>0)[:,1]]
print("Items bought by other customer:")
items_bought_by_customerB
```

```python colab={"base_uri": "https://localhost:8080/", "height": 717} id="8DL_RNZjACSR" outputId="aee04f04-66a2-4ab3-e86f-a9c0e77a783c"
items_to_recommend_to_customerB= set(items_bought_by_customerA) - set(items_bought_by_customerB)
print("Items to Recommend to customer B:")
data_full.loc[data_full['productid'].isin(items_to_recommend_to_customerB),['productid', 'name']].drop_duplicates().set_index('productid')
```

<!-- #region id="H7oVusIFAfOo" -->
> Tip: For Item-item similarity, take the transpose of session-item matrix and repeat the same steps. 
<!-- #endregion -->
