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

<!-- #region id="ZzWWk2KpZgJo" -->
# GroupBy E-commerce Recommender System with Implicit data
> Recommendation system using collaborative filtering for implicit data on e-commerce customer journey. Capstone project for Machine Learning Engineer Bootcamp at FourthBrain. It is an industry sponsored project from GroupBy, an e-commerce start up that aims to create highly converting and relevant site experience to maximize revenue through e-commerce channels. Also benchmarking movielens and cosmetics datasets.

- toc: true
- badges: true
- comments: true
- categories: [Cosmetics, Retail, ECommerce, LightFM, Implicit, Flask, FrontEnd, API, Movie, ElasticBeanStalk]
- author: "<a href='https://github.com/tjeng/recommendation'>Janice Tjeng</a>"
- image:
<!-- #endregion -->

<!-- #region id="kFZ1snpBYXyb" -->
## Introduction
<!-- #endregion -->

<!-- #region id="sgl5qWRM2UP-" -->
Industry sponsored project from GroupBy, an e-commerce start up that aims to create highly converting and relevant site experience to maximize revenue through e-commerce channels
<!-- #endregion -->

<!-- #region id="R5JZrRbu2Y0O" -->
Online shopping offers millions of items for users to choose from, but with limited attention span and limited real estate space, it becomes necessary to handle information overload. Recommendation system can solve the problem by ranking and recommending top items based on users' preference. It is also reported that 35% of Amazon’s revenue comes from its recommendation engine.
<!-- #endregion -->

<!-- #region id="olMQwU9n5jx9" -->
## Setup
<!-- #endregion -->

```python id="6-h1KKJJ5iL0"
!pip install lightfm
```

```python id="LubpezQS5mk7"
import json
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

# lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
```

<!-- #region id="rdEStuMC2bN-" -->
## Data
GroupBy has provided its internal data (raw data not shared) that contains customer product interaction data where there is information on the interaction with an item (search, view, add to cart, or purchase), as well as product brand, and price collected over 1 day.

A user-item matrix is created that contains 44,588 users and 17,480 products, with 1M interactions. The interaction is a binary 1 or 0 that indicates whether the user purchases the item or not. This is a form of implicit feedback where user's preference for an item is indirectly inferred from behavior patterns versus explicit feedback where the user gives a rating of the product.

The dataset is randomly split based on interactions into train and test sets, 80% and 20% respectively. All users and items are in train and test sets but 20% of the interactions are masked from the training set. There might not be any interaction for a user in the train set and the items recommended is compared to the items purchased in the test set to evaluate model performance.
<!-- #endregion -->

```python id="LuvhccbyyVph"
!mkdir data
!cd data && wget https://github.com/sparsh-ai/recommendation/raw/main/recdeployment/recapp/data/interactions.npz
!cd data && wget https://github.com/sparsh-ai/recommendation/raw/main/recdeployment/recapp/data/item_dictionary.json
!cd data && wget https://github.com/sparsh-ai/recommendation/raw/main/recdeployment/recapp/data/item_mapping.json
!cd data && wget https://github.com/sparsh-ai/recommendation/raw/main/recdeployment/recapp/data/user_mapping.json
```

<!-- #region id="wqCwlXIX57Hd" -->
## Data loading
<!-- #endregion -->

```python id="af46bd00" outputId="7890bf56-bdcc-438f-8bde-4acfc52aaeb7"
data = [json.loads(line) for line in open('raw_data.json', 'r')]
df = pd.json_normalize(data)
df.customerVisitorId.nunique()
```

```python id="ce5ff415" outputId="75e278a7-4e68-452c-91f0-d506bef36369"
df.columns
```

<!-- #region id="rZ7xb7nl6iAq" -->
## Preprocessing
<!-- #endregion -->

```python id="4e2b7356" outputId="e572dd8e-3a98-4054-993a-de3cb8185cad"
cols = ['customerVisitorId', 'customerSessionId','sessionStartTime', 'sessionEndTime', 'customerSessionNumber']
hits = pd.json_normalize(data, record_path=['hits', 'product'], meta=cols)
hits[hits.customerVisitorId=='cki8pfl1y000139ebdeezn6pn']
```

```python id="LP9_xiUH6DCA" outputId="a612d27d-f18e-4272-9652-ce44fea7c592"
hits.collection.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 576} id="1403d256" outputId="975e1a51-087b-4102-f6b9-e0686d8eefb9"
all_cols = df.columns[df.columns.str.contains('totals')].tolist() + cols
df_merged = df[all_cols].merge(hits, how='left', on=cols)
df_merged[df_merged.customerVisitorId=='cki8pfl1y000139ebdeezn6pn'][['ID', 'name', 'price', 'totals.totalOrders', 'totals.totalOrderQty', 'customerSessionId', 'customerSessionNumber', 'totals.uniqueOrders', 'totals.totalViewProducts', 'totals.totalAddToCarts']]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 352} id="OfHlBXMWt9UW" outputId="32b3e7ea-126f-41a7-9c1c-76cfb792fc55"
plt.hist(hits.price)
```

<!-- #region id="Q7oq9zQU6phh" -->
**Filter price**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4z5CRnnltqH_" outputId="4f1cadbf-c900-4472-9100-42db334b2d1b"
df_merged2 = df_merged[(df_merged.price>=11) & (df_merged.price<=24)] 
md = np.median(hits[~hits.price.isnull()].price)
len(hits['price'])
```

```python id="iWP4C8zo6DCH" outputId="379d73cc-cb52-48ba-d1fc-19dacbe11fba"
df_rm_dup = df_merged.drop_duplicates(['customerVisitorId', 'ID', 'price', 'customerSessionId'])
df_rm_dup.collection.value_counts(normalize=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZChVgmS3tXV_" outputId="7c33af4e-2c09-4a2e-c41f-d98f7f2a8a3c"
df_rm_dup[['ID', 'price', 'collection', 'name']].info()
```

```python id="n5osuQBI6DCJ" outputId="16cec085-df87-476e-a787-c65043704442"
df_rm_dup[df_rm_dup.name.isnull()].ID
```

<!-- #region id="BpzgolAo616V" -->
**Create dictionary of item ID, name, price, collection**
<!-- #endregion -->

```python id="wYp6OlzX6DCM" outputId="99cf8979-277f-4933-9d36-f4c48cbb912a"
mult_collection = df_rm_dup.groupby('ID')['collection'].nunique().sort_values(ascending=False).reset_index()
mult_collection
```

```python id="krC1mlvo6DCM" outputId="04b72556-b09d-43e7-a71d-ae2515c6c29e"
hits[hits.ID=='SW923'].collection.unique()
```

```python id="r5lX62Tg6DCN"
dic = {}
for i in range(len(df_rm_dup)):
    prod = df_rm_dup.iloc[i]
    if prod.ID not in dic.keys() and str(prod.sku) != 'nan':
        dic[prod.ID] = {}
        dic[prod.ID]['name'] = prod['name']
        dic[prod.ID]['collection'] = prod.collection
        dic[prod.ID]['price'] = prod.price
```

```python id="eb8da79b"
df_rm_dup['totals.totalOrderQty'] = df_rm_dup['totals.totalOrderQty'].fillna(0)
df_rm_dup['totals.totalOrderQty'] = df_rm_dup['totals.totalOrderQty'].astype(int)
num_purchase = df_rm_dup.groupby(['customerVisitorId', 'ID']).agg({'totals.totalOrderQty':'sum', 'price':'mean', 'customerSessionId':'nunique', 'collection':'first'})
num_purchase = num_purchase.reset_index().rename(columns={"totals.totalOrderQty":"numPurchase"})
num_purchase['Purchase'] = num_purchase.numPurchase.apply(lambda x: 1 if x!=0 else 0)
```

```python id="uvq3LY606DCP" outputId="b7fb84ef-8b84-4b5c-c12f-ebef9ea84e98"
cust_collection = num_purchase.groupby("customerVisitorId")['collection'].apply(list).reset_index()
cust_collection['contains_swansonhealth'] = cust_collection['collection'].apply(lambda x: 'swansonhealthproduction' not in x)
num_purchase['price'] = np.where(num_purchase.price.isnull(), md, num_purchase.price)
num_purchase['price_category'] = np.where(num_purchase.price>=94, 'very expensive', np.where((num_purchase.price>= 48) & (num_purchase.price<94), 'expensive', np.where((num_purchase.price>= 24) & (num_purchase.price<48), 'medium', 'cheap')))
num_purchase.head()
```

<!-- #region id="VSU0uyl87JdU" -->
**Analysis of product price category and number of sessions with regards to purchase**
<!-- #endregion -->

```python id="H1MxMVQX6DCQ" outputId="a49e85f9-e2b1-4977-8f72-142f36efafca"
num_purchase[num_purchase.Purchase==1].groupby('price_category')['customerSessionId'].mean()
```

```python id="ahP-GEeQ6DCR" outputId="2f077cc3-98f4-402e-df1f-da98eaaae84b"
plt.scatter(num_purchase[(num_purchase.collection=='swansonhealthproduction')].price_category, num_purchase[num_purchase.collection=='swansonhealthproduction'].customerSessionId)
plt.xlabel('price category')
plt.ylabel('number of sessions')
plt.title('Groupby Data Filter by Category')
```

```python id="cbw4N-f56DCR" outputId="8c009af6-d06f-4434-c832-12248299d1b2"
num_purchase[(num_purchase.Purchase==1) & (num_purchase.price>=11) & (num_purchase.price<=24)].ID.nunique()
```

```python id="EHIOpdDt6DCR" outputId="2d5485eb-c159-4fda-9969-8546fbe260ba"
num_purchase.groupby('price_category')['ID'].count()
```

```python id="1aRjcXM-6DCS" outputId="d3bf9ea1-7049-4614-f2b4-f687425dada3"
plt.scatter(num_purchase[num_purchase.Purchase==0].customerSessionId, num_purchase[num_purchase.Purchase==0].price)
```

<!-- #region id="59deb013" -->
**Splitting data into train and test sets**
<!-- #endregion -->

```python id="ILxcT9WK9tqi"
def random_split(data, user_id, item_id, metric, item_features=None):
  dataset = Dataset()
  dataset.fit(data[user_id], data[item_id], item_features=item_features)
  num_users, num_items = dataset.interactions_shape()
  print('Num users: {}, num_items {}.'.format(num_users, num_items))
  dataset.fit_partial(users=data[user_id], 
                     items=data[item_id],
                     item_features=item_features)
  fe = data.to_dict(orient='records')
  (interactions, weights) = dataset.build_interactions((x[user_id], x[item_id], x[metric]) for x in fe)
  (train, test) = random_train_test_split(interactions=interactions, test_percentage=0.2, random_state=2)
  print('Num values: {}'.format(len(interactions.data)))  
  return dataset, train, test, interactions
```

```python id="YBbk53L26DCU"
def calc_sparsity(data, user_id, item_id, metric):
    item_user_matrix = pd.DataFrame(data.pivot_table(index=[user_id], columns=item_id, values=metric))
    item_user_matrix.fillna(0,inplace=True)
    print('sparsity in data:',(1.0 - (np.count_nonzero(item_user_matrix) / float(item_user_matrix.size))))
```

```python id="_2rArOfo6DCV" outputId="b5602ca6-cf36-4ec4-bc6f-eaf32635a5fd"
dataset, train_g, test_g, interactions = random_split(num_purchase, 'customerVisitorId', 'ID', 'Purchase')
```

```python id="DhESoEjw6DCV" outputId="d10e04b0-f8d1-4997-a708-3bf3d3f65ded"
train_g.shape
```

```python id="9Y_Y4iYU6DCW" outputId="929cc599-dbda-4de0-cf48-7adaccbe1e43"
test_g.shape
```

<!-- #region id="SAwn6KM16DCW" -->
**Create dictionary to map user and item index**
<!-- #endregion -->

```python id="O_i9Ydkz6DCX"
user_id_map, user_features, item_id_map, item_features = dataset.mapping()
```

```python id="qA0K7wMz6DCX"
item_id_map_rev = {int(v):k for k,v in item_id_map.items()}
```

```python id="FEA3fekB6DCX"
user_id_map_rev = {int(v):k for k,v in user_id_map.items()}
```

```python id="_FXg3ROj6DCY" outputId="49f20686-35bd-4a7d-a11a-3213aae93957"
train_g
```

```python id="L_x0oGwG6DCY" outputId="3b8986cf-ebb1-447d-8c3d-28ece5491482"
test_g
```

```python id="V1lkOujn6DCa" outputId="cb207304-b9da-4a56-d32b-5584010ecb0f"
calc_sparsity(num_purchase, 'customerVisitorId', 'ID', 'Purchase')
```

<!-- #region id="DlJhdITO2f9_" -->
## Algorithm
<!-- #endregion -->

<!-- #region id="3pkjTmxg2jEL" -->
Collaborative Filtering Matrix Factorization (MF) is used where an mxn matrix (m: number of users, n: number of items) is decomposed into mxk user factor and kxn item factor. The factors are multiplied to get the score for an item for a particular user. The figure below illustrates the matrix factorization. k represents the number of latent factors, or dimensions.

LightFM package link is used to perform MF. A common loss function for MF is mean square error where the gradient updates to minimize the difference between actual and predicted scores. Such technique is common for explicit feedback data. For this dataset that consists of implicit feedback, ranking of items, where a purchased item ranks higher than a non-purchased item, are of interest. Hence, weighted average ranking pairwise (WARP) loss link is used as the loss function. For each user, there is a pair of positive and negative items. A positive item indicates that the user has purchased the item and a negative item indicates that the user has not purchased the item. The loss function only updates when the rank of a negative item exceeds that of a positive item. This approximates a form of active learning that yields a more informative gradient update, where the model samples the number of negative items Q times until the rank of a negative item exceeds that of a positive item. The loss function is described below.
<!-- #endregion -->

<!-- #region id="YOkiHBKJ2nvA" -->
<!-- #endregion -->

<!-- #region id="cgOJeaf1pxHk" -->
## LightFM model training
<!-- #endregion -->

```python id="XJJvHKlt-oGe"
def model_fit_eval(loss, train, test, n_components, k_items, k=5, n=10, item_features=None):
  model = LightFM(no_components=n_components, loss=loss, item_alpha=1e-6, random_state=2, k=k, n=n)
  %time model.fit(train, epochs=10, num_threads=3, item_features=item_features)
  test_auc = auc_score(model, test, train_interactions=train, num_threads=3, item_features=item_features).mean()
  print('Collaborative filtering test AUC: %s' % test_auc)
  prec = precision_at_k(model, test, train_interactions=train, num_threads=3, k=k_items, item_features=item_features).mean()
  print('Collaborative filtering test precision: %s' % prec)
  recal = recall_at_k(model, test, train_interactions=train, num_threads=3, k=k_items, item_features=item_features).mean()
  print('Collaborative filtering test recall: %s' % recal)
  return model
```

```python id="-9JfCZ_x6DCc" outputId="bc52ec9d-c131-4979-cff9-703c744b0331"
model_g = model_fit_eval('warp', train_g, test_g, 128, 5)
```

<!-- #region id="yOTpLBZj2pcp" -->
## Results
<!-- #endregion -->

<!-- #region id="a3t_hIb72sJ3" -->
The important parameter to tune is k, the number of dimensions. The table below shows model performance with different values of k:

| K | Metric | Value |
| - | -:| ---:|
| 32 | AUC | 0.87 |
| 64 | AUC | 0.88 |
| 128 | AUC | 0.89 |
| 192 | AUC | 0.89 |
| 32 | Precision@5 | 0.38 |
| 64 | Precision@5 | 0.32 |
| 128 | Precision@5 | 0.40 |
| 192 | Precision@5 | 0.41 |
| 32 | Recall@5 | 0.22 |
| 64 | Recall@5 | 0.27 |
| 128 | Recall@5 | 0.30 |
| 192 | Recall@5 | 0.31 |

- 128 dimension is selected as model performance starts to plateau.
- AUC measures the probability that a randomly chosen positive example has a higher score than a randomly chosen negative example.
- Precision@5 measures the fraction of items bought out of the 5 recommended items
- Recall@5 measures the number of items bought in the top 5 recommendations divided by the total number of items bought

<!-- #endregion -->

<!-- #region id="Hq6baCTj6DCe" -->
## Recommending items
<!-- #endregion -->

```python id="PxEOI-1O6DCe"
def similar_recommendation(model, interactions, user_id, user_dikt, 
                               item_dikt, item_dikt_rev, product_dict, threshold = 0, number_rec_items = 5):

    #Function to produce user recommendations

    n_items = len(item_dikt.keys())
    user_x = user_dikt[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    scores.index = item_dikt.keys()
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    user_item_scores = interactions.toarray()[user_x]
    item_index = np.where(user_item_scores>0)[0]
    known_items = pd.DataFrame({ 'items': [item_dikt_rev[i] for i in item_index], 'scores':user_item_scores[item_index]}).sort_values('scores', ascending=False)
    
    scores = [x for x in scores if x not in known_items['items'].values]
    score_list = scores[0:number_rec_items]

    print("Items that were liked by the User:")
    counter = 1
    for i in range(len(known_items)):
        item = known_items.loc[i]['items']
        if str(product_dict[item]['name']) != 'nan':
            print(str(counter) + '- ' + str(product_dict[item]['name']) + ', ' + str(product_dict[item]['price']) + ', ' + product_dict[item]['collection'])
        else:
            print(str(counter) + '- ' + str(product_dict[item]['collection']))
        counter+=1

    print("\n Recommended Items:")
    counter = 1
    for i in score_list:
        if str(product_dict[i]['name']) != 'nan':
            print(str(counter) + '- ' + product_dict[i]['name'] + ', ' + str(product_dict[i]['price']) + ', ' + product_dict[i]["collection"]) 
        else:
            print(str(counter) + '- ' + product_dict[i]['collection'])
        counter+=1
#     return score_list
```

```python id="n5vJEnY-6DCf" outputId="12e1de1e-e5e8-4538-cc55-588589e82a00"
similar_recommendation(model, interactions, 'cjo06esrd00013gblo1n3haw7', user_id_map, 
                               item_id_map, item_id_map_rev, product_dic, threshold = 0, number_rec_items = 5)
```

```python id="Dkc3g33D6DCf" outputId="0bac844e-19fd-49f2-9341-d03bf94e4763"
similar_recommendation(model, interactions, 'cjo09x89u00013gb80mn9xb57', user_id_map, 
                               item_id_map, item_id_map_rev, product_dic, threshold = 0, number_rec_items = 5)
```

```python id="FuI07ewb6DCg" outputId="a544798f-f11d-41d1-cf42-91efdaabb8f0"
similar_recommendation(model_g, 'cjo6507qf00013ac37tbxxu8z', user_id_map, 
                               item_id_map, item_id_map_rev, dic, threshold = 0, number_rec_items = 5)
```

```python id="cL9VyalZ6DCh"
def item_emdedding_distance_matrix(model, item_dikt):

    # Function to create item-item distance embedding matrix
    df_item_norm_sparse = csr_matrix(model.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = item_dikt.keys()
    item_emdedding_distance_matrix.index = item_dikt.keys()
    return item_emdedding_distance_matrix

def also_bought_recommendation(item_emdedding_distance_matrix, item_id, 
                             product_dict, n_items = 5):

    # Function to create item-item recommendation
    recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \
                                  sort_values(ascending = False).head(n_items+1). \
                                  index[1:n_items+1]))
    
    print("Item of interest:")
    print(str(product_dict[item_id]['name']) + ', ' + str(product_dict[item_id]['price']) + ', ' + product_dict[item_id]['collection'])
    print("\n")
    print("Items that are frequently bought together:")
    counter = 1
    for i in recommended_items:
        if i in product_dict.keys():
            print(str(counter) + '- ' + str(product_dict[i]['name']) + ', ' + str(product_dict[i]['price']) + ', ' + product_dict[i]['collection'])
        else:
            print(str(counter) + '- ' + 'swansonhealthproduct')
        counter+=1
    #return recommended_items
```

```python id="C-AHn5hw6DCi"
item_dist = item_emdedding_distance_matrix(model, item_id_map)
```

```python id="h4Wyshem6DCi" outputId="c009893c-22d1-4420-ec8b-b09449d1db6f"
also_bought_recommendation(item_dist, 'SW1113', dic)
```

```python id="rub6WhT_6DCj" outputId="b980a441-1639-4b22-c58f-239f617b90c3"
also_bought_recommendation(item_dist, 'SWD015', dic)
```

<!-- #region id="eDsL8AB-V8L3" -->
## API
<!-- #endregion -->

<!-- #region id="qGEQLIep4AwT" -->
For a demo, the model is served as an API using Flask. The Flask app is deployed on AWS Elastic Beanstalk, with the following url, http://recapp.eba-u3jbujfw.us-east-1.elasticbeanstalk.com/. To use the app, enter a user id. If the user id exists in the existing database of users, top 5 items will be recommended based on the user's purchasing history. If the user is a new user, the top 5 most popular items (purchased by most users) will be displayed instead. Due to proprietary information, user id to test will not be provided.
<!-- #endregion -->

<!-- #region id="Iw6cqx3w4T_p" -->
> Note: This EBS is currently active but might not work in future. So we included screenshots along with backend HTML for reference.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LXJ1DM8LyrSL" outputId="e13a27d3-cc5d-474a-d193-7ce73cefbe70"
%%writefile index.html
<!DOCTYPE html>
<html >
<head>
  <meta charset="UTF-8">
  <title>Product Recommendation</title>
  <link href='https://fonts.googleapis.com/css?family=Pacifico' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Arimo' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Hind:300' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300' rel='stylesheet' type='text/css'>
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
 <div class="login">
	<h1> Enter a User Id</h1>
     <!-- Main Input For Receiving Query to our ML -->
    <form action="{{ url_for('recommend')}}"method="post">
    	<input type="text" name="userid" placeholder="User Id"/>
        <button type="submit" class="btn">Recommend</button>
    </form>
   <br>
   <br>
 </div>
</body>
</html>
```

<!-- #region id="XoRFVizR1mGd" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xjtCEstyy4T9" outputId="1d3a79a4-6a6c-4683-eb8d-5872352490e7"
%%writefile recommendation.html
<!DOCTYPE html>
<html >
<head>
  <meta charset="UTF-8">
  <title>Recommending top 5 items: </title>
  <link href='https://fonts.googleapis.com/css?family=Pacifico' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Arimo' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Hind:300' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300' rel='stylesheet' type='text/css'>
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
 <div style="background-color:white;">
	<h2>Items purchased:</h2>
        {% for answ in items%}
                <p style="text-align: left; font-family: normal; font-size: 
 16px;"> {{ answ  }} </p>
           {% endfor %}
	<h2>Items recommended:</h2>
        {% for answ in rec_items%}
                <p style="text-align: left; font-family: normal; font-size: 
 16px;"> {{ answ  }} </p>
           {% endfor %}  
 </div>
</body>
</html>
```

```python colab={"base_uri": "https://localhost:8080/"} id="31Rdhpf1zAIJ" outputId="0312a349-219d-4739-ddb4-924efff1611b"
%%writefile topitems.html
<!DOCTYPE html>
<html >
<head>
  <meta charset="UTF-8">
  <title>Recommending top 5 items: </title>
  <link href='https://fonts.googleapis.com/css?family=Pacifico' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Arimo' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Hind:300' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300' rel='stylesheet' type='text/css'>
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
 <div style="background-color:white;">
	<h2>Items recommended:</h2>
        {% for answ in rec_items%}
                <p style="text-align: left; font-family: normal; font-size: 
 16px;"> {{ answ  }} </p>
           {% endfor %}  
 </div>
</body>
</html>
```

<!-- #region id="0315HhYq1t7E" -->
<!-- #endregion -->

<!-- #region id="oPvg-Z3OYKS0" -->
### Recommender app
<!-- #endregion -->

```python id="0V-rFmbQzLcO"
class Recommendation:
    def __init__(self):
        with open("user_mapping.json", "r") as uf:
            self.user_dic = json.load(uf)
        with open("item_mapping.json") as imf:
            self.item_dic = json.load(imf)
        with open("item_dictionary.json") as pf:
            self.product_dic = json.load(pf)
        with open("./recapp/static/model/recommender_model_alldata.pkl", 'rb') as f:
            self.model = pickle.load(f)
        self.interactions = sparse.load_npz("interactions.npz")
        self.item_rev_dic = {v:k for k,v in self.item_dic.items()}

    def similar_recommendation(self, user_id, threshold = 0, number_rec_items = 5):
        #Function to produce user recommendations
        n_items = len(self.item_dic.keys())
        user_x = self.user_dic[user_id]
        scores = pd.Series(self.model.predict(user_x,np.arange(n_items)))
        scores.index = self.item_dic.keys()
        scores = list(pd.Series(scores.sort_values(ascending=False).index))
        user_item_scores = self.interactions.toarray()[user_x]
        item_index = np.where(user_item_scores>0)[0]
        known_items = pd.DataFrame({ 'items': [self.item_rev_dic[i] for i in item_index], 'scores':user_item_scores[item_index]}).sort_values('scores', ascending=False)
        
        scores = [x for x in scores if x not in known_items['items'].values]
        score_list = scores[0:number_rec_items]

        items_output = []
        #rint("Items that were liked by the User:")
        counter = 1
        for i in range(len(known_items)):
            item = known_items.loc[i]['items']
            if str(self.product_dic[item]['name']) != 'nan':
                items_output.append(str(counter) + '- ' + str(self.product_dic[item]['name']) + ', ' + str(self.product_dic[item]['price']) + ', ' + self.product_dic[item]['collection'])
            else:
                items_output.append(str(counter) + '- ' + str(self.product_dic[item]['collection']))
            counter+=1

        rec_output = []
        #print("\n Recommended Items:")
        counter = 1
        for i in score_list:
            if str(self.product_dic[i]['name']) != 'nan':
                rec_output.append(str(counter) + '- ' + self.product_dic[i]['name'] + ', ' + str(self.product_dic[i]['price']) + ', ' + self.product_dic[i]["collection"]) 
            else:
                rec_output.append(str(counter) + '- ' + self.product_dic[i]['collection'])
            counter+=1
        return items_output, rec_output
		
    def top_n_items(self, n = 5):
    	np_item = self.interactions.toarray().sum(axis=0)
    	df = pd.DataFrame({'num_purchase':np_item})
    	top_n_df = df.sort_values(['num_purchase'], ascending=False).head(n).index
    	top_n = [self.item_rev_dic[i] for i in top_n_df]
    	products = []
    	counter = 1
    	for i in top_n:
    		if str(self.product_dic[i]['name']) != 'nan':
    			products.append(str(counter) + '- ' + self.product_dic[i]['name'] + ', ' + str(self.product_dic[i]['price']) + ', ' + self.product_dic[i]["collection"]) 
    		else:
    			products.append(str(counter) + '- ' + self.product_dic[i]['collection'])
    		counter+=1
    	return products

    def item_embedding_distance_matrix(self):
    #     Function to create item-item distance embedding matrix
        df_item_norm_sparse = csr_matrix(self.model.item_embeddings)
        similarities = cosine_similarity(df_item_norm_sparse)
        item_emdedding_distance_matrix = pd.DataFrame(similarities)
        item_emdedding_distance_matrix.columns = self.item_dic.keys()
        item_emdedding_distance_matrix.index = self.item_dic.keys()
        return item_emdedding_distance_matrix

    def also_bought_recommendation(self, item_emdedding_distance_matrix, item_id, n_items = 5):
    #     Function to create item-item recommendation
        recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \
                                      sort_values(ascending = False).head(n_items+1). \
                                      index[1:n_items+1]))
        
        #print("Item of interest:")
        item_interest = (str(self.product_dic[item_id]['name']) + ', ' + str(self.product_dic[item_id]['price']) + ', ' + self.product_dic[item_id]['collection'])
        #print("\n")
        #print("Items that are frequently bought together:")
        item_rec = []
        counter = 1
        for i in recommended_items:
            if i in self.product_dic.keys():
                item_rec.append(str(counter) + '- ' + str(self.product_dic[i]['name']) + ', ' + str(self.product_dic[i]['price']) + ', ' + self.product_dic[i]['collection'])
            else:
                item_rec.append(str(counter) + '- ' + 'swansonhealthproduct')
            counter+=1
        return item_interest, item_rec
```

<!-- #region id="bVyEZEcMYH7y" -->
### Routes
<!-- #endregion -->

```python id="cXZvNb4czTUq"
from flask import Flask, request, jsonify, render_template
from flask import current_app as app

r = Recommendation()

#item_distance_matrix = r.item_embedding_distance_matrix()
# SW1113, SWD015, cjo6507qf00013ac37tbxxu8z, cjo09x89u00013gb80mn9xb57, cjo06esrd00013gblo1n3haw7

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
	id = [str(x) for x in request.form.values()]
	userid = id[0]
	if userid in r.user_dic.keys():
		item_output, rec_output = r.similar_recommendation(userid)
		return render_template('recommendation.html', items=item_output, rec_items=rec_output)
	else:
		rec_output = r.top_n_items()
		return render_template('topitems.html', rec_items = rec_output)
```

<!-- #region id="RVh2IayR8ofq" -->
## Benchmark other data
<!-- #endregion -->

<!-- #region id="bHVMZ86mCBID" -->
### Movielens
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XL4QQPdV8ESB" outputId="981ee44d-66dc-4b17-8698-13ad380ef819"
from lightfm.datasets import fetch_movielens

data = fetch_movielens(min_rating=5.0)

model = LightFM(loss='warp', no_components=192, item_alpha=1e-6, random_state=2)
model.fit(data['train'], epochs=10, num_threads=3)

test_auc = auc_score(model, data['test'], train_interactions=data['train'], num_threads=3).mean()
print('Collaborative filtering test AUC: %s' % test_auc)
prec = precision_at_k(model, data['test'], train_interactions=data['train'], num_threads=3, k=5).mean()
print('Collaborative filtering test precision: %s' % prec)
recal = recall_at_k(model, data['test'], train_interactions=data['train'], num_threads=3, k=5).mean()
print('Collaborative filtering test recall: %s' % recal)
```

<!-- #region id="y-DYf0IZRiYg" -->
### Cosmetics
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Y3OjJUM4OOdO" outputId="22cd839f-fd57-462c-8d22-0af07befbbc1"
!pip install -q git+https://github.com/sparsh-ai/recochef.git

from recochef.datasets.cosmetics import Cosmetics

cdata = Cosmetics()
cdata_list = [cdata.load_interactions(f"cdata{idx}.parquet.gzip", chunk=idx) for idx in range(1,6)]
cosmetics_data = pd.concat(cdata_list, axis=0, ignore_index=True)
cosmetics_data.info()
```

<!-- #region id="f9sVsAogWJie" -->
> Note: Taking a 10% sample (~2 million) for faster processing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LbE0rZo0Wnm_" outputId="a0a7b1c8-02af-4d65-b903-838311b73901"
cosmetics_data_sample = cosmetics_data.sample(frac=0.1)
cosmetics_data_sample.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="JabrtOWtQnsr" outputId="3e62d5c7-ea25-4405-ad1e-526268babc71"
cosmetics_data_2 = cosmetics_data_sample.copy()
cosmetics_data_2 = cosmetics_data_2.sort_values('TIMESTAMP').groupby(['USERID', 'ITEMID'])

_data = cosmetics_data_2['EVENTTYPE'].value_counts().unstack(fill_value = 0).rename(columns={'view':'NumTimesViewed',
                                                                                             'cart':'NumTimesCarted',
                                                                                             'purchase':'NumTimesPurchased',
                                                                                             'remove_from_cart':'NumTimesRemoved'})
_data = _data.reset_index()
_data['Purchase'] = _data.NumTimesPurchased.apply(lambda x: 1 if x!=0 else 0)

_data2 = cosmetics_data_2.agg({'PRICE':'mean', 'SESSIONID':'nunique', 'CATEGORYID':'first'}).reset_index()

_data = _data.merge(_data2, how='inner', on=['USERID', 'ITEMID'])

_data['PRICECATEGORY'] = np.where(_data.PRICE>=99, 'very expensive',
                                  np.where((_data.PRICE >= 66) & (_data.PRICE < 99), 'expensive',
                                           np.where((_data.PRICE >= 33) & (_data.PRICE<66), 'medium', 'cheap')))

_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="CMJcE255Tr8v" outputId="a0031ef9-3231-4719-c2ae-97086abe800a"
dataset_k, train, test, int_k = random_split(_data, 'USERID', 'ITEMID', 'Purchase')
```

```python colab={"base_uri": "https://localhost:8080/"} id="fEEyH_tvVJEJ" outputId="f5464dab-df40-465e-f895-5ecfc81f9f74"
model_c = model_fit_eval('warp', train, test, 32, 5)
```

```python id="Tc8OCE4bcRnY" outputId="3888f5a6-0a90-4d71-a523-8d2ec63aebbc"
model_c = model_fit_eval('warp-kos', train, test, 128, k=7, n=20)
```
