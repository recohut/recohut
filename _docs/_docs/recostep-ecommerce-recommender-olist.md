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

<!-- #region id="a7b80794" -->
# Recommendation systems using Olist dataset
> Olist e-commerce dataset to build simple recommender systems

- toc: true
- badges: true
- comments: true
- categories: [ecommerce]
- image: 
<!-- #endregion -->

```python id="5d2c5778"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import re
from textblob import TextBlob
```

```python id="7EV9Nrncva6X"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d olistbr/brazilian-ecommerce
```

```python colab={"base_uri": "https://localhost:8080/"} id="Z9X0umLBvomC" outputId="d6aa51f3-1720-416c-fbae-b0730e0a600e"
!unzip brazilian-ecommerce.zip
```

```python id="8defdd68"
products =  pd.read_csv('olist_products_dataset.csv', usecols=['product_id','product_category_name'])

orders = pd.read_csv('olist_orders_dataset.csv',usecols = ['order_id','customer_id'])

prod = pd.read_csv('olist_order_items_dataset.csv',usecols = ['product_id','order_id'])

customers = pd.read_csv('olist_customers_dataset.csv',usecols = ['customer_id','customer_zip_code_prefix','customer_city'])

location = pd.read_csv('olist_geolocation_dataset.csv', usecols = ['geolocation_zip_code_prefix'])

reviews = pd.read_csv('olist_order_reviews_dataset.csv',usecols = ['order_id','review_score','review_comment_message'])
```

```python id="f24b05e9"
def describe_data(df):
  print("Data Types:")
  print(df.dtypes)
  print("\n\nRows and Columns:")
  print(df.shape)
  print("\n\nColumn Names:")
  display(df.columns.tolist())
  print("\n\nNull values")
  print(df.isnull().sum())
```

```python id="fa9dc282" colab={"base_uri": "https://localhost:8080/", "height": 323} outputId="8ba1e204-58df-4905-cc77-ed625eae8802"
describe_data(products)
```

```python id="25ecca66" colab={"base_uri": "https://localhost:8080/", "height": 357} outputId="bf6c8fdd-f0ec-4444-eafe-e6f8a9ae7ad5"
describe_data(customers)
```

```python id="7a6c9b0b" colab={"base_uri": "https://localhost:8080/", "height": 357} outputId="c6db002b-6a6d-419d-c7ae-b9760f9db81c"
describe_data(reviews)
```

```python id="c4c55008" colab={"base_uri": "https://localhost:8080/", "height": 323} outputId="cae40be8-52b1-4677-9531-2957eb8a898f"
describe_data(orders)
```

```python id="36f09457" colab={"base_uri": "https://localhost:8080/", "height": 289} outputId="9ce0b2ea-b81c-465b-9ff2-96952e0726d4"
describe_data(location)
```

```python id="b54b453a" colab={"base_uri": "https://localhost:8080/", "height": 323} outputId="3ca324e3-46c0-4aa8-c497-ce45e64e028e"
describe_data(prod)
```

```python id="a408f031" colab={"base_uri": "https://localhost:8080/", "height": 325} outputId="46d3e060-6ae7-44e4-f583-a466a793e07a"
plt.rc("font", size=15)
reviews.review_score.value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('system1.png', bbox_inches='tight')
plt.show()
```

<!-- #region id="6943fa53" -->
### Popularity based recommender
<!-- #endregion -->

```python id="6523011e"
comb_product = pd.merge(products, prod, on ='product_id')
```

```python id="cca67fa4" colab={"base_uri": "https://localhost:8080/", "height": 309} outputId="c553b285-7357-4c31-dc9d-249e5af88fea"
comb_product_review = pd.merge(comb_product,reviews, on = 'order_id')
comb_product_review.head(5)
```

```python id="bd6c4149" colab={"base_uri": "https://localhost:8080/", "height": 235} outputId="8e02c94d-3660-4ec4-ee63-53434a2e0a80"
review_count = pd.DataFrame(comb_product_review.groupby('product_category_name')['review_score'].count())
review_count.sort_values('review_score', ascending=False).head()
```

<!-- #region id="18e79bd8" -->
#### Conclusion : These are the top 5 products with highest review score and so we can recommend these ones. Best recommended technique when completely new user visits an e-commerce site, that site will not have any past history

<!-- #endregion -->

<!-- #region id="6a22dfd9" -->
### KNN collaborative method

The collaborative filtering algorithm uses “User Behavior” for recommending items.

kNN is a machine learning algorithm to find clusters of similar users based on ratings, and make predictions using the average rating of top-k nearest neighbors. For example, we first present ratings in a matrix with the matrix having one row for each item and one column for each user location

<!-- #endregion -->

```python id="c6ad9a97"
comb_product_review = comb_product_review[comb_product_review.review_score >= 3]
```

```python id="7df199ad"
prod_order_review = pd.merge(comb_product_review, orders , on = 'order_id')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ZFyy0BxXyxah" outputId="12a89260-835d-40f1-d078-5962c299dfc3"
customers.head()
```

```python id="81a742a1"
cust_location = pd.merge(customers, location, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix')
```

```python id="7cf09b56" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="55ed7664-8a0b-4245-870d-9d5346d8b755"
cust_location.head(5)
```

```python id="9eeec582"
cust_prod_location = pd.merge (prod_order_review, cust_location, on = 'customer_id' )
```

```python id="42508597" colab={"base_uri": "https://localhost:8080/"} outputId="0449595d-9cf9-4fe9-8ddd-48f9d981f7df"
print(cust_prod_location['review_score'].quantile(np.arange(.9,1,.01)))
```

```python id="4bf95057"
from scipy.sparse import csr_matrix

location_user_rating = cust_prod_location.drop_duplicates(['customer_zip_code_prefix', 'product_category_name'])

location_user_rating_pivot = location_user_rating.pivot(index = 'product_category_name', columns = 'customer_zip_code_prefix', values = 'review_score').fillna(0)

location_user_rating_matrix = csr_matrix(location_user_rating_pivot.values)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="mBkaAS1M0H7O" outputId="7884f30a-338b-479c-8765-e2e420019491"
location_user_rating_pivot.sample(10).iloc[:,:10]
```

```python id="ab057802" colab={"base_uri": "https://localhost:8080/"} outputId="f29581a5-57f6-4de1-fbf4-968769b596d7"
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')

model_knn.fit(location_user_rating_matrix)
```

```python id="2677cc21" colab={"base_uri": "https://localhost:8080/"} outputId="bef27ef0-2d28-475b-c3d5-4d83e68a502d"
query_index = np.random.choice(location_user_rating_pivot.shape[0])
print(query_index)
distances, indices = model_knn.kneighbors(location_user_rating_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
```

```python id="049229c9" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="4130fb50-2d9e-4f53-c168-b59a1c56e5a2"
location_user_rating_pivot.index[query_index]
```

```python id="fa75a03b" colab={"base_uri": "https://localhost:8080/"} outputId="17145e67-bbfd-4707-8781-d868bc5bc6b7"
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(location_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, location_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
```

<!-- #region id="e3d91cd3" -->
### Recommendation based on sentiment analysis of Review message
<!-- #endregion -->

```python id="c66b0ef5" colab={"base_uri": "https://localhost:8080/", "height": 490} outputId="14879ece-22cd-4a59-f41c-306ff5c69b5d"
prod_order_review
```

```python id="299131dc"
prod_order_review.dropna(subset = ["review_comment_message"], inplace=True)
```

```python id="de868c15"
final = prod_order_review[['product_category_name','review_comment_message']]
```

```python id="61d43e5f"
pd.set_option('mode.chained_assignment', None)

# Convert to list
final['data'] = final['review_comment_message'].to_list()
```

```python id="064a84ff"
# Pre-processing steps for data

final['data'] = [re.sub('\s*@\s*\s?', ' ', str(sent)) for sent in final['data']]

final['data'] = [re.sub('\?', ' ', str(sent)) for sent in final['data']]

final['data'] = [re.sub('\_', ' ', str(sent)) for sent in final['data']]

final['data'] = [re.sub('@"[\d-]"', ' ', str(sent)) for sent in final['data']]

# Remove new line characters
final['data'] = [re.sub('\s+', ' ', str(sent)) for sent in final['data']]

# Remove distracting single quotes
final['data'] = [re.sub("\'", " ", str(sent)) for sent in final['data']]

#Converting into lowercase
final['data']=final['data'].str.lower()

```

```python id="vQ5QZOPL2-Vf"
bloblist_desc = list()

df_comments= final['data'].astype(str)

for row in df_comments:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    df_comments_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['review_comment_message','sentiment','polarity'])
 
def f(df_comments_polarity_desc):
    if df_comments_polarity_desc['sentiment'] > 0:
        val = "Positive"
    elif df_comments_polarity_desc['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

df_comments_polarity_desc['Sentiment_Type'] = df_comments_polarity_desc.apply(f, axis=1)
```

```python id="bc91a307" colab={"base_uri": "https://localhost:8080/", "height": 335} outputId="cd98625c-e51a-4ad7-fe74-af7108b2acc3"
plt.figure(figsize=(5,5))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data = df_comments_polarity_desc)
```

```python id="8c26fb32" colab={"base_uri": "https://localhost:8080/", "height": 419} outputId="a159ace5-2053-4b8e-9b54-ed673fd3405e"
df_comments_polarity_desc
```

```python id="f820e1c1"
review_analytics = pd.merge(final, df_comments_polarity_desc, on = 'review_comment_message')
```

```python id="55b0ae53"
review_analytics = review_analytics[review_analytics.Sentiment_Type == 'Positive']
```

```python id="7c7eda7b" colab={"base_uri": "https://localhost:8080/"} outputId="84863c04-cc3d-4b6e-8ece-b1175d7de65b"
review_analytics.product_category_name.unique()
```

```python id="080da7cc" colab={"base_uri": "https://localhost:8080/"} outputId="e606ae2a-1180-41a7-8c1a-3fe41a066d1e"
len(review_analytics.product_category_name.unique())
```

<!-- #region id="0f4ddf70" -->
####  Conclusion - These are the products recommended based on sentiments 
<!-- #endregion -->

<!-- #region id="f11dca94" -->
### Future Ideas:

1. Recommendation system using Hybrid approach

2. System can be particularly built using data of customers, products at different location 

3. Can also try system with customer payment history
    
<!-- #endregion -->
