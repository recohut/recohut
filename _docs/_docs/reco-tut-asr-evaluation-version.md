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

```python id="86MgMsi_GD70" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628258438065, "user_tz": -330, "elapsed": 3134, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6f05d446-2c70-49f3-da8e-1c8a32b9643a"
import os
project_name = "reco-tut-asr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="f8773e69" executionInfo={"status": "ok", "timestamp": 1628258649870, "user_tz": -330, "elapsed": 811, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="FfnYpw9lsSlV" executionInfo={"status": "ok", "timestamp": 1628259070315, "user_tz": -330, "elapsed": 770, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f9829d06-057c-49d6-97f3-71f303d8e3cd"
items = pd.read_csv('./data/silver/items.csv')
items.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 253} id="CwNn1S6OseTn" executionInfo={"status": "ok", "timestamp": 1628259072791, "user_tz": -330, "elapsed": 466, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d3f4906b-c633-4c72-d1de-075a3413cd15"
actual_ratings = pd.read_csv('./data/silver/ratings.csv')
actual_ratings.head()
```

```python id="OcWjqLZlyrhB" executionInfo={"status": "ok", "timestamp": 1628259081026, "user_tz": -330, "elapsed": 663, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
cbf = pd.read_csv('./data/gold/cbf.csv')
item_item = pd.read_csv('./data/gold/item-item.csv')
user_user = pd.read_csv('./data/gold/user-user.csv')
pers_bias = pd.read_csv('./data/gold/pers-bias.csv')
mf = pd.read_csv('./data/gold/mf.csv')
```

```python id="_FTl6xQdsEmG" executionInfo={"status": "ok", "timestamp": 1628259083334, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# preprocess
cbf = cbf.apply(lambda col: col.apply(lambda elem: str(elem).replace(',', '.'))).astype(float)
user_user = user_user.apply(lambda col: col.apply(lambda elem: str(elem).replace(',', '.'))).astype(float)
item_item = item_item.apply(lambda col: col.apply(lambda elem: str(elem).replace(',', '.'))).astype(float)
mf = mf.apply(lambda col: col.apply(lambda elem: str(elem).replace(',', '.'))).astype(float)
pers_bias = pers_bias.apply(lambda col: col.apply(lambda elem: str(elem).replace(',', '.'))).astype(float)
```

```python id="9de630de" executionInfo={"status": "ok", "timestamp": 1628259085227, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
recs = [cbf, item_item, user_user, pers_bias, mf]
recs_names = ['cbf', 'item_item', 'user_user', 'pers_bias', 'mf']
```

<!-- #region id="du1uGAsbx-fw" -->
## Metrics
<!-- #endregion -->

```python id="63NhJNhyx4KQ" executionInfo={"status": "ok", "timestamp": 1628259088092, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_ratings(user_id):
    user_ratings = ratings[user_id]
    actual_ratings = user_ratings[~np.isnan(user_ratings)]
    return actual_ratings


def get_top_n(user_id, n):
    top_n = {}
    for rec, rec_name in zip(recs, recs_names):
        top_n_items = rec[user_id].argsort().sort_values()[:n].index.values
        top_n[rec_name] = top_n_items
    return top_n


def get_popular_items(n):
    pop_percentages = ratings.copy()
    pop_percentages['popularity'] = ratings.apply(lambda row: np.sum(~np.isnan(row))-1, axis=1)/len(ratings.columns[1::])
    pop_percentages = pop_percentages.sort_values(by = 'popularity', ascending=False)
    return pop_percentages.item.values[:n]


def get_rmse(user_id):   
    user_ratings = get_ratings(user_id)
    rmse = {}
    for rec, rec_name in zip(recs, recs_names):
        predicted_ratings = rec.loc[user_ratings.index, user_id]
        temp = np.sqrt(np.average((predicted_ratings - user_ratings)**2))
        rmse[rec_name] = temp
    return rmse


def get_precision_at_n(user_id, n):
    top_n = get_top_n(user_id, n)
    user_ratings = get_ratings(user_id).index.values
    precisions = {}
    for rec, rec_name in zip(recs, recs_names):
        temp = np.sum(np.isin(top_n[rec_name], user_ratings))/n
        precisions[rec_name] = temp
    return precisions


# We will use the "FullCat" column in the items catalog to determine the product diversity in the recommendations.
# The recommender with a high number of distinct product categories in its recommendations is said to be product-diverse
def get_product_diversity(user_id, n):
    top_n = get_top_n(user_id, n)
    product_diversity = {}
    for rec_name in top_n:
        categories = items.loc[top_n[rec_name]][['FullCat']].values
        categories = set([item for sublist in categories for item in sublist])
        product_diversity[rec_name] = len(categories)
    return product_diversity


# We will use the "Price" column in the items catalog to determine cost diversity in the recommendations.
# The recommender with a high standard deviation in the cost across all its recommendations is said to be cost-diverse
def get_cost_diversity(user_id, n):
    top_n = get_top_n(user_id,n)
    cost_diversity = {}
    for rec_name in top_n:
        std_dev = np.std(items.loc[top_n[rec_name]][['Price']].values)
        cost_diversity[rec_name] = std_dev
    return cost_diversity


# We will use inverse popularity as a measure of serendipity.
# The recommender with least number of recommendations on the "most popular" list, will be called most serendipitous
def get_serendipity(user_id, n):
    top_n = get_top_n(user_id,n)
    popular_items = get_popular_items(20)
    serendipity = {}
    for rec, rec_name in zip(recs, recs_names):
        popularity = np.sum(np.isin(top_n[rec_name],popular_items))
        if int(popularity) == 0:
            serendipity[rec_name] = 1
        else:
            serendipity[rec_name] = 1/popularity
    return serendipity
```

```python id="Gp0DegGhx66S" executionInfo={"status": "ok", "timestamp": 1628259095684, "user_tz": -330, "elapsed": 7598, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
avg_metrics = {}
for name in recs_names: 
    avg_metrics[name] = {"rmse": [], "precision_at_n": [], "product_diversity": [], "cost_diversity": [], "serendipity": []}

for user_id in ratings.columns:
    if user_id == 'item':
        continue
    user_id = str(user_id)
    rmse = get_rmse(user_id)
    precision_at_n = get_precision_at_n(user_id, 10)
    product_diversity = get_product_diversity(user_id, 10)
    cost_diversity = get_cost_diversity(user_id, 10)
    serendipity = get_serendipity(user_id, 10)
    for key in avg_metrics:
        rec_name = avg_metrics[key]
        rec_name['rmse'].append(rmse[key])
        rec_name['precision_at_n'].append(precision_at_n[key])
        rec_name['product_diversity'].append(product_diversity[key])
        rec_name['cost_diversity'].append(cost_diversity[key])
        rec_name['serendipity'].append(serendipity[key])

# The Price for certain items is not available. Also rmse for certain users is turning out to be NaN.
# Ignoring nans in the average metric calculation for now. So basically narrowing down the evaluation to users who have
# rated atleast one item and items for which the price is known.
for key in avg_metrics:
    rec_name = avg_metrics[key]
    for metric in rec_name:
        temp = rec_name[metric]
        temp = [x for x in temp if not np.isnan(x)]
        rec_name[metric] = sum(temp) / len(temp)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3QY49QhNyA_Y" executionInfo={"status": "ok", "timestamp": 1628259095687, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c445c8e8-320e-4eae-b93a-4ba1659057a7"
avg_metrics
```

<!-- #region id="def2d993" -->
## Hybridization
<!-- #endregion -->

```python id="6fa69df3" executionInfo={"status": "ok", "timestamp": 1628259105317, "user_tz": -330, "elapsed": 441, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Creating a dataframe with ratings from all algorithms and user_ratings as ground truth
users = []
items = []
user_ratings = []
cbf_ratings = []
user_user_ratings = []
item_item_ratings = []
mf_ratings = []
pers_bias_ratings = []

for user_id in ratings.columns:
    if user_id == 'item':
        continue
    user_id = str(user_id)
    true_ratings = get_ratings(user_id)
    user_ratings.extend(true_ratings.values)
    users.extend([user_id]*len(true_ratings))
    items.extend(ratings.loc[true_ratings.index].item.values)
    cbf_ratings.extend(cbf.loc[true_ratings.index, user_id].values)
    item_item_ratings.extend(item_item.loc[true_ratings.index, user_id].values)
    user_user_ratings.extend(user_user.loc[true_ratings.index, user_id].values)
    pers_bias_ratings.extend(pers_bias.loc[true_ratings.index, user_id].values)
    mf_ratings.extend(mf.loc[true_ratings.index, user_id].values)
    
df = pd.DataFrame({'user': users, 'item': items,'true_rating': user_ratings, 'cbf':cbf_ratings, 'item_item':item_item_ratings, 'user_user': user_user_ratings, 'pers_bias':pers_bias_ratings, 'mf':mf_ratings})
```

```python id="9b41eb5b" executionInfo={"status": "ok", "timestamp": 1628259107917, "user_tz": -330, "elapsed": 459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df = df.dropna()
```

```python id="9cef2129" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1628259108641, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5db3bf94-0d2f-4dab-a5e7-3bfa385cba69"
df.head()
```

<!-- #region id="e44690b8" -->
### Linear Combination
<!-- #endregion -->

```python id="a2291f18" executionInfo={"status": "ok", "timestamp": 1628259141254, "user_tz": -330, "elapsed": 654, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
clf = LinearRegression()
```

```python id="81df9b09" executionInfo={"status": "ok", "timestamp": 1628259141777, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Split data in 80-20 train and test sets
train = df[0:(int(0.8*len(df)))]
test = df[(int(0.8*len(df)))::]
```

```python id="af0518e6" executionInfo={"status": "ok", "timestamp": 1628259143511, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train_data = train.drop(['user', 'item','true_rating'], axis=1)
train_labels = train.true_rating.values
model = clf.fit(train_data, train_labels)
```

```python id="8d56dd06" executionInfo={"status": "ok", "timestamp": 1628259144072, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
test_data = test.drop(['user', 'item','true_rating'], axis=1)
test_labels = test.true_rating.values
predictions = model.predict(test_data)
```

```python id="3eb03663" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628259149601, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f950bb82-3bb7-41eb-a875-6f2b4b03e59d"
# Avg RMSE predictions
avg_rmse = np.sqrt(np.average((predictions - test_labels)**2))
avg_rmse
```

<!-- #region id="8f39db75" -->
#### Top 5 for three users
<!-- #endregion -->

```python id="fb2cafde" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628259168931, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1c0b58f4-f73c-4707-c750-bb0603de5064"
# Pick three users
users = random.sample(list(ratings.columns[1::]), 3)
print(users)
```

```python id="049812b9" executionInfo={"status": "ok", "timestamp": 1628259171508, "user_tz": -330, "elapsed": 733, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train_data = df.drop(['user', 'item','true_rating'], axis=1)
train_labels = df.true_rating.values
model = clf.fit(train_data, train_labels)
```

```python id="39296114"
top_5 = {}
for user in users:
    df_preds = df[df.user == user]
    preds = model.predict(df_preds.drop(['user', 'item','true_rating'], axis=1))
    df_preds['predictions'] = preds
    top_5_items = list(df_preds.sort_values(by=['predictions'], ascending=False)[:5].item.values)
    top_5[user] = top_5_items
```

```python id="94c2ddb6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628259176882, "user_tz": -330, "elapsed": 470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="43d5e81e-3af3-4a9c-9ffe-a0492a06062c"
top_5
```

<!-- #region id="e7d1dd5e" -->
### Non-linear Combination 
<!-- #endregion -->

<!-- #region id="971242e7" -->
For a non-linear combination of the algorithms, we'll use the DecisionTreeRegressor method in scikitlearn
<!-- #endregion -->

```python id="57b3fc3b" executionInfo={"status": "ok", "timestamp": 1628259194855, "user_tz": -330, "elapsed": 748, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
clf = DecisionTreeRegressor()
```

```python id="a6492796" executionInfo={"status": "ok", "timestamp": 1628259195549, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Split data in 80-20 train and test sets
train = df[0:(int(0.8*len(df)))]
test = df[(int(0.8*len(df)))::]
```

```python id="d62bfaa5" executionInfo={"status": "ok", "timestamp": 1628259196941, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train_data = train.drop(['user', 'item','true_rating'], axis=1)
train_labels = train.true_rating.values
model = clf.fit(train_data, train_labels)
```

```python id="a75936a3" executionInfo={"status": "ok", "timestamp": 1628259196943, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
test_data = test.drop(['user', 'item','true_rating'], axis=1)
test_labels = test.true_rating.values
predictions = model.predict(test_data)
```

```python id="09b21a15" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628259202501, "user_tz": -330, "elapsed": 732, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bf14cb51-4985-479e-82de-525708e23a81"
# Avg RMSE predictions
avg_rmse = np.sqrt(np.average((predictions - test_labels)**2))
avg_rmse
```

<!-- #region id="9811b1ef" -->
#### Top-5 for 3 users
<!-- #endregion -->

```python id="5a00f88b" executionInfo={"status": "ok", "timestamp": 1628259215228, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Using the same users as above to compare across the same users
users = ['3430', '112', '1817']
```

```python id="3a79acb3" executionInfo={"status": "ok", "timestamp": 1628259215689, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train_data = df.drop(['user', 'item','true_rating'], axis=1)
train_labels = df.true_rating.values
model = clf.fit(train_data, train_labels)
```

```python id="7bac243a"
top_5 = {}
for user in users:
    df_preds = df[df.user == user]
    preds = model.predict(df_preds.drop(['user', 'item','true_rating'], axis=1))
    df_preds['predictions'] = preds
    top_5_items = list(df_preds.sort_values(by=['predictions'], ascending=False)[:5].item.values)
    top_5[user] = top_5_items
```

```python id="05d348de" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628259222068, "user_tz": -330, "elapsed": 811, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f43e9acd-3a87-4dd1-f04c-7e49685141e0"
top_5
```

<!-- #region id="38996984" -->
## Different recommenders based on user type
<!-- #endregion -->

<!-- #region id="80460aa6" -->
This hybridization techniques aims to create separate recomemnder strategies for two separate scenarios- one where users end up on the Nile-River.com landing page via banner ads for school products and other where users arrive at the landing page via endoresements for office products. For the first scenario, we'll pick a 3:2 ratio of school (inexpensive) products vs. office (expensive) products and the reverse for the second scenario i.e. 2:3 ratio of school to office products. Here we will show the evaluate only for the first scenario.
<!-- #endregion -->

```python id="7dcc6543" executionInfo={"status": "ok", "timestamp": 1628259313109, "user_tz": -330, "elapsed": 462, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Determine threshold to label an item cheap or expensive- let's set this as the third quantile of the price list
# This is assuming office products are mostly in the expensive bracket
items = pd.read_csv('./data/silver/items.csv') # df converted to list in processing above, so loading back
prices = items.Price.values
price_threshold = np.percentile([x for x in prices if not np.isnan(x)], 75)
```

<!-- #region id="04ac9c55" -->
### Performance
<!-- #endregion -->

```python id="5560bb4d" executionInfo={"status": "ok", "timestamp": 1628259317013, "user_tz": -330, "elapsed": 737, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_precision_at_n(user_id, top_n):
    user_ratings = get_ratings(user_id).index.values
    precision_at_n = np.sum(np.isin(top_n, user_ratings))/ len(top_n)
    return precision_at_n
```

```python id="035c22b2" executionInfo={"status": "ok", "timestamp": 1628259317614, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_cost_diversity(top_n):
    std_dev = np.std(items.loc[top_n][['Price']].values)
    return std_dev
```

```python id="13a1459e" executionInfo={"status": "ok", "timestamp": 1628259317616, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_product_diversity(top_n):
    categories = items.loc[top_n][['FullCat']].values
    categories = set([item for sublist in categories for item in sublist])
    return len(categories)
```

```python id="9142de97" executionInfo={"status": "ok", "timestamp": 1628259317617, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_serendipity(top_n):
    popular_items = get_popular_items(20)
    popularity = np.sum(np.isin(top_n,popular_items))
    if int(popularity) == 0:
        serendipity = 1
    else:
        serendipity = 1/popularity
    return serendipity
```

```python id="71c3f6c7" executionInfo={"status": "ok", "timestamp": 1628259317618, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# To pick which items to finally recommend, let's assume that all the items in the top-5 for each recommender are 
# equally relevant. We can potentially include some ranking based selection to pick item that are more relavant AND fit the
# cost criteria. For now, we'll pick at random since we're assuming all items are equally relevant.
def get_mixed_recs(user_id, n, n_cheap, n_exp):
    top_n_overall_items = []
    top_n_overall_prices = []
    mixed_recs = [] 
    for rec, rec_name in zip(recs, recs_names):
        top_n_items = rec[user_id].argsort().sort_values()[:n].index.values
        top_n_prices = items.loc[top_n_items][['Price']].values
        top_n_overall_items.extend(top_n_items)
        top_n_overall_prices.extend(top_n_prices)
    top_dict = dict(zip(top_n_overall_items, top_n_overall_prices))
    top_cheap = dict(filter(lambda elem: elem[1] <= price_threshold, top_dict.items())).keys()
    top_exp = dict(filter(lambda elem: elem[1] > price_threshold, top_dict.items())).keys()
    mixed_recs = random.sample(list(top_cheap), n_cheap) + random.sample(list(top_exp), n_exp)
    return mixed_recs
```

```python id="bee18803" executionInfo={"status": "ok", "timestamp": 1628259325590, "user_tz": -330, "elapsed": 6675, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
avg_metrics = {"precision_at_n": [], "product_diversity": [], "cost_diversity": [], "serendipity": []}
for user_id in ratings.columns:
    if user_id == 'item':
        continue
    user_id = str(user_id)
    top_5 = get_mixed_recs(user_id, 5, 3, 2)
    avg_metrics["precision_at_n"].append(get_precision_at_n(user_id, top_5))
    avg_metrics["cost_diversity"].append(get_cost_diversity(top_5))
    avg_metrics["product_diversity"].append(get_product_diversity(top_5))
    avg_metrics["serendipity"].append(get_serendipity(top_5))

for metric in avg_metrics:
    temp = avg_metrics[metric]
    temp = [x for x in temp if not np.isnan(x)]
    avg_metrics[metric] = sum(temp) / len(temp)
```

```python id="b08a6ad3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628259325592, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ddb0ad4b-7dd9-4268-d5e1-d01a7afa0d10"
avg_metrics
```

<!-- #region id="e8efdad3" -->
### Top-5 for three users
<!-- #endregion -->

```python id="b9b5f512" executionInfo={"status": "ok", "timestamp": 1628259326398, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Assuming all three users ended up on the landing pagee through scenario 1 i.e. banner ads for school products
users = ['3430', '112', '1817']
```

```python id="fba7a9ad" executionInfo={"status": "ok", "timestamp": 1628259326842, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
top_5 = {}
for user_id in users:
    # For office products
    # top_5[user_id] = get_mixed_recs(user_id, 5, 2, 3)
    # For school products
    top_5[user_id] = list(ratings.loc[get_mixed_recs(user_id, 5, 3, 2)].item.values)
```

```python id="f6fa453d" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628259326843, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="15e7fa6a-a3a5-4535-a6a3-895b506ff0e3"
top_5
```

<!-- #region id="3dc699c9" -->
## Switching hybridization
<!-- #endregion -->

<!-- #region id="f7c776ad" -->
We will not be implementing this hybridizaton as such, but we will explore whether or not the strategy of using content based filtering for new users (users with fewer/no ratings) or items with less ratings is even reasonable for this dataset. For this, let's begin with visualizing the number of ratings for the users in the dataset.
<!-- #endregion -->

```python id="cd1e3e91" executionInfo={"status": "ok", "timestamp": 1628259339315, "user_tz": -330, "elapsed": 1225, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
item_ratings = ratings.apply(lambda row: np.sum(~np.isnan(row))-1, axis=1)
```

```python id="485cd4cf" colab={"base_uri": "https://localhost:8080/", "height": 296} executionInfo={"status": "ok", "timestamp": 1628259339316, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ffda4bce-e8c9-4863-f269-23dd88be4c55"
plt.hist(item_ratings)
plt.xlabel("Number of ratings")
plt.ylabel("number of items")
```

```python id="16da3538" executionInfo={"status": "ok", "timestamp": 1628259340771, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Number of items with < 10 ratings
count_less_than_10 = np.count_nonzero(item_ratings<10)/len(item_ratings)*100
```

```python id="6884b455" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628259341647, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="922d0569-2219-4211-b3df-6654c54a6322"
count_less_than_10
```

```python id="1cea4d00" executionInfo={"status": "ok", "timestamp": 1628259341651, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
user_ratings = []
for user_id in ratings.columns:
    if user_id == 'item':
        continue
    user_id = str(user_id)
    user_ratings.append(len(get_ratings(user_id)))
```

```python id="7dbde101" colab={"base_uri": "https://localhost:8080/", "height": 296} executionInfo={"status": "ok", "timestamp": 1628259344085, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e5f9bebc-285d-4a1c-ce92-f9d7e8693fea"
plt.hist(user_ratings)
plt.xlabel("Number of ratings")
plt.ylabel("number of users")
```

```python id="ce4d9ea9" executionInfo={"status": "ok", "timestamp": 1628259344086, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Number of users with < 10 ratings
count_less_than_10 = np.count_nonzero(np.array(user_ratings)<10)/len(user_ratings)*100
```

```python id="618605d0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628259344087, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e9720b20-f485-4b30-f48f-5598619f96c5"
count_less_than_10
```
