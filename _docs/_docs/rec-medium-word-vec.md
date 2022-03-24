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
    language: python
    name: python3
---

<!-- #region id="QU13Xg9n0bSM" -->
# Retail Product Recommendations using word2vec
> Creating a system that automatically recommends a certain number of products to the consumers on an E-commerce website based on the past purchase behavior of the consumers.

- toc: true
- badges: true
- comments: true
- categories: [sequence, retail]
- image: 
<!-- #endregion -->

<!-- #region id="VwgbuHwVtshy" -->
A person involved in sports-related activities might have an online buying pattern similar to this:
<!-- #endregion -->

<!-- #region id="NQLbbKfmtn_O" -->
<!-- #endregion -->

<!-- #region id="2tYh4SrWtynJ" -->
If we can represent each of these products by a vector, then we can easily find similar products. So, if a user is checking out a product online, then we can easily recommend him/her similar products by using the vector similarity score between the products.
<!-- #endregion -->

```python executionInfo={"elapsed": 1427, "status": "ok", "timestamp": 1619252390068, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="W0wI5j74nc9W"
#hide
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
%matplotlib inline

import warnings;
warnings.filterwarnings('ignore')
```

<!-- #region id="cDi5Gu8Ou7bb" -->
## Data gathering and understanding
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3274, "status": "ok", "timestamp": 1619252391927, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="TlW0pTzGniGo" outputId="0392f779-4c42-4e9e-b096-51ee6487b53d"
#hide-output
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 45709, "status": "ok", "timestamp": 1619252434373, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="z_Kt8wtZnjRm" outputId="e30cdfd6-fd16-48ab-fa22-283fdb3d2578"
df = pd.read_excel('Online Retail.xlsx')
df.head()
```

<!-- #region id="KJHexOy0oPHl" -->
Given below is the description of the fields in this dataset:

1. __InvoiceNo:__ Invoice number, a unique number assigned to each transaction.

2. __StockCode:__ Product/item code. a unique number assigned to each distinct product.

3. __Description:__ Product description

4. __Quantity:__ The quantities of each product per transaction.

5. __InvoiceDate:__ Invoice Date and time. The day and time when each transaction was generated.

6. __CustomerID:__ Customer number, a unique number assigned to each customer.
<!-- #endregion -->

<!-- #region id="h8BwXsF5ox--" -->
## Data Preprocessing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 45703, "status": "ok", "timestamp": 1619252434375, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="F58y_bU6nypT" outputId="26f4b722-e131-46af-9dde-3cf388986882"
# check for missing values
df.isnull().sum()
```

<!-- #region id="JM04nhUAot7Y" -->
Since we have sufficient data, we will drop all the rows with missing values.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 45696, "status": "ok", "timestamp": 1619252434376, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="uqEjaaTKorZ4" outputId="061df95f-dfbf-4fda-fd6c-f3f329df87ed"
# remove missing values
df.dropna(inplace=True)

# again check missing values
df.isnull().sum()
```

```python executionInfo={"elapsed": 46353, "status": "ok", "timestamp": 1619252435036, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="RUib7vkCpd1E"
# Convert the StockCode to string datatype
df['StockCode']= df['StockCode'].astype(str)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 46347, "status": "ok", "timestamp": 1619252435036, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="wPI7YWkQo6wu" outputId="e751a6d9-6ba2-42a7-8a4f-1aaee616f2b4"
# Check out the number of unique customers in our dataset
customers = df["CustomerID"].unique().tolist()
len(customers)
```

<!-- #region id="ED5y4TDxpOfL" -->
There are 4,372 customers in our dataset. For each of these customers we will extract their buying history. In other words, we can have 4,372 sequences of purchases.
<!-- #endregion -->

<!-- #region id="Aogi1piGvEGl" -->
## Data Preparation
<!-- #endregion -->

<!-- #region id="F_WFU-rTvJ_l" -->
It is a good practice to set aside a small part of the dataset for validation purpose. Therefore, we will use data of 90% of the customers to create word2vec embeddings. Let's split the data.
<!-- #endregion -->

```python executionInfo={"elapsed": 46345, "status": "ok", "timestamp": 1619252435037, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="WxWaZx3zpPFW"
# shuffle customer ID's
random.shuffle(customers)

# extract 90% of customer ID's
customers_train = [customers[i] for i in range(round(0.9*len(customers)))]

# split data into train and validation set
train_df = df[df['CustomerID'].isin(customers_train)]
validation_df = df[~df['CustomerID'].isin(customers_train)]
```

<!-- #region id="ul94yT2cqJ38" -->
Let's create sequences of purchases made by the customers in the dataset for both the train and validation set.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 51786, "status": "ok", "timestamp": 1619252440487, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="uhIwWEe-qGwK" outputId="6de33b11-1871-4fea-e60e-56e4cb03db7a"
# list to capture purchase history of the customers
purchases_train = []

# populate the list with the product codes
for i in tqdm(customers_train):
    temp = train_df[train_df["CustomerID"] == i]["StockCode"].tolist()
    purchases_train.append(temp)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 52224, "status": "ok", "timestamp": 1619252440935, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="VGT9oyVeqhky" outputId="8198f6f0-9649-4214-ef55-20576f4a6e9f"
# list to capture purchase history of the customers
purchases_val = []

# populate the list with the product codes
for i in tqdm(validation_df['CustomerID'].unique()):
    temp = validation_df[validation_df["CustomerID"] == i]["StockCode"].tolist()
    purchases_val.append(temp)
```

<!-- #region id="AgDLwI_4q4Fm" -->
## Build word2vec Embeddings for Products
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 106693, "status": "ok", "timestamp": 1619252495414, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="rr_tHmmuqu24" outputId="dcedddc7-d410-4d0c-bd3a-79f6022f42ef"
# train word2vec model
model = Word2Vec(window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(purchases_train, progress_per=200)

model.train(purchases_train, total_examples = model.corpus_count, 
            epochs=10, report_delay=1)
```

```python executionInfo={"elapsed": 106690, "status": "ok", "timestamp": 1619252495414, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="_CwTzmNqq_lQ"
# save word2vec model
model.save("word2vec_2.model")
```

<!-- #region id="HBbudJ7hrDw0" -->
As we do not plan to train the model any further, we are calling init_sims(), which will make the model much more memory-efficient
<!-- #endregion -->

```python executionInfo={"elapsed": 106689, "status": "ok", "timestamp": 1619252495415, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="UGYK8p1xrAJy"
model.init_sims(replace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 106680, "status": "ok", "timestamp": 1619252495416, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="M_AbUfMOrGsI" outputId="66d6d641-2dfb-4129-b27c-6eeafffd07a2"
print(model)
```

<!-- #region id="HKFT2SECrMY1" -->
Now we will extract the vectors of all the words in our vocabulary and store it in one place for easy access
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 106664, "status": "ok", "timestamp": 1619252495416, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="odn6f3rorG7T" outputId="f8c06bf9-1ba4-48f9-e491-1fb2369fdadc"
# extract all vectors
X = model[model.wv.vocab]

X.shape
```

<!-- #region id="VQr5YRjIrTC2" -->
## Visualize word2vec Embeddings
<!-- #endregion -->

<!-- #region id="EwW0xbwkrVoB" -->
It is always quite helpful to visualize the embeddings that you have created. Over here we have 100 dimensional embeddings. We can't even visualize 4 dimensions let alone 100. Therefore, we are going to reduce the dimensions of the product embeddings from 100 to 2 by using the UMAP algorithm, it is used for dimensionality reduction.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 109415, "status": "ok", "timestamp": 1619252498177, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="A7XmYq1Eragv" outputId="9dfed0d8-4a6c-4cd5-bf14-9bbe0e8248f2"
#hide
!pip install umap-learn
```

```python colab={"base_uri": "https://localhost:8080/", "height": 54} executionInfo={"elapsed": 150522, "status": "ok", "timestamp": 1619252539293, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Y8wKkbRQrQAy" outputId="0b928341-1790-411d-fb91-a824f3c05264"
#collapse
import umap

cluster_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
                              n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(10,9))
plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], s=3, cmap='Spectral');
```

<!-- #region id="wNQ9hotsr1eC" -->
Every dot in this plot is a product. As you can see, there are several tiny clusters of these datapoints. These are groups of similar products.
<!-- #endregion -->

<!-- #region id="5_KBjh2Jr-Qg" -->
## Generate and validate recommendations
<!-- #endregion -->

<!-- #region id="he6azpQysC_M" -->
We are finally ready with the word2vec embeddings for every product in our online retail dataset. Now our next step is to suggest similar products for a certain product or a product's vector. 

Let's first create a product-ID and product-description dictionary to easily map a product's description to its ID and vice versa.
<!-- #endregion -->

```python executionInfo={"elapsed": 150520, "status": "ok", "timestamp": 1619252539294, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="sckVfOMHrYHp"
products = train_df[["StockCode", "Description"]]

# remove duplicates
products.drop_duplicates(inplace=True, subset='StockCode', keep="last")

# create product-ID and product-description dictionary
products_dict = products.groupby('StockCode')['Description'].apply(list).to_dict()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 150514, "status": "ok", "timestamp": 1619252539296, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Ldj0ew7UsOiw" outputId="e03b87e5-8f3c-466e-a85c-ec004746efdc"
# test the dictionary
products_dict['84029E']
```

<!-- #region id="eneoWxrwsSjt" -->
We have defined the function below. It will take a product's vector (n) as input and return top 6 similar products.
<!-- #endregion -->

```python executionInfo={"elapsed": 150514, "status": "ok", "timestamp": 1619252539298, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="TmgZr0c2sPuz"
#hide
def similar_products(v, n = 6):
    
    # extract most similar products for the input vector
    ms = model.similar_by_vector(v, topn= n+1)[1:]
    
    # extract name and similarity score of the similar products
    new_ms = []
    for j in ms:
        pair = (products_dict[j[0]][0], j[1])
        new_ms.append(pair)
        
    return new_ms        
```

<!-- #region id="H3ygmMQHsZRA" -->
Let's try out our function by passing the vector of the product '90019A' ('SILVER M.O.P ORBIT BRACELET')
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 150506, "status": "ok", "timestamp": 1619252539299, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="pWnP50EVsWEj" outputId="bec2fe21-2259-4ae6-a30d-5cb4667d53d7"
similar_products(model['90019A'])
```

<!-- #region id="aH-2nt_ishkM" -->
Cool! The results are pretty relevant and match well with the input product. However, this output is based on the vector of a single product only. What if we want recommend a user products based on the multiple purchases he or she has made in the past?

One simple solution is to take average of all the vectors of the products he has bought so far and use this resultant vector to find similar products. For that we will use the function below that takes in a list of product ID's and gives out a 100 dimensional vector which is mean of vectors of the products in the input list.
<!-- #endregion -->

```python executionInfo={"elapsed": 150504, "status": "ok", "timestamp": 1619252539299, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="MqtpgpqFsauG"
#collapse
def aggregate_vectors(products):
    product_vec = []
    for i in products:
        try:
            product_vec.append(model[i])
        except KeyError:
            continue
        
    return np.mean(product_vec, axis=0)
```

<!-- #region id="mXPqS4h0sojc" -->
If you can recall, we have already created a separate list of purchase sequences for validation purpose. Now let's make use of that.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 151381, "status": "ok", "timestamp": 1619252540183, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="KfI_RrLZsn8W" outputId="929f549e-1621-474d-9c96-f4012a1aa5f9"
#hide
len(purchases_val[0])
```

<!-- #region id="nuaqWGa4ssZr" -->
The length of the first list of products purchased by a user is 314. We will pass this products' sequence of the validation set to the function aggregate_vectors.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 151376, "status": "ok", "timestamp": 1619252540184, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="OdurKB3ysp6c" outputId="c2635b4c-584c-47a0-e436-f4175fc87dad"
#hide
aggregate_vectors(purchases_val[0]).shape
```

<!-- #region id="gwaXOQSus9ya" -->
Well, the function has returned an array of 100 dimension. It means the function is working fine. Now we can use this result to get the most similar products. Let's do it.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 151371, "status": "ok", "timestamp": 1619252540186, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="PKTVsoc3s7ZK" outputId="ec1f2e74-74bc-4035-e8f6-b01b95b35f4d"
similar_products(aggregate_vectors(purchases_val[0]))
```

<!-- #region id="MRsPrkMotDaZ" -->
As it turns out, our system has recommended 6 products based on the entire purchase history of a user. Moreover, if you want to get products suggestions based on the last few purchases only then also you can use the same set of functions.

Below we are giving only the last 10 products purchased as input.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 151364, "status": "ok", "timestamp": 1619252540187, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="OUqyW1-Ns_u2" outputId="c903bdf6-ac73-4a53-eaa6-8987a67cc8fb"
similar_products(aggregate_vectors(purchases_val[0][-10:]))
```

<!-- #region id="59iEesnTzNHq" -->
## References

- [https://www.analyticsvidhya.com/blog/2019/07/how-to-build-recommendation-system-word2vec-python/](https://www.analyticsvidhya.com/blog/2019/07/how-to-build-recommendation-system-word2vec-python/)
- [https://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/](https://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/)
- [https://www.analyticsinsight.net/building-recommendation-system-using-item2vec/](https://www.analyticsinsight.net/building-recommendation-system-using-item2vec/)
- [https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484](https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484)
- [https://capablemachine.com/2020/06/23/word-embedding/](https://capablemachine.com/2020/06/23/word-embedding/)
<!-- #endregion -->

```python executionInfo={"elapsed": 151363, "status": "ok", "timestamp": 1619252540188, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Q--xnW3dtIZy"

```
