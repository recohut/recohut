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

# Two-Stage Recommendation System

<!-- #region id="GnGNXUrcQLd6" -->
### Why 2 levels?
- Classical classification models (lightgbm) often perform better than recommender models (als, lightfm)
- There is a lot of data, a lot of predictions (# items * # users) -> lightgbm cannot cope with such a volume
- But recommender models do the trick!
- We select top-N (200) candidates using a simple model (als) -> re-rank them with a complex model (lightgbm) and select top-k (10).
<!-- #endregion -->

<!-- #region id="yudjgeYvRS1C" -->
## Candidate selection
- We generate top-k candidates
- The quality of candidates is measured through recall @ k
- recall @ k shows what proportion of the purchased goods we were able to identify (recommend) with our model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ZFm18V7vRiDt" outputId="15077fed-2289-45bc-fdf2-0e070cb17a5c"
!pip install -q implicit
```

```python id="4e-ywTDOUgDX"
%reload_ext autoreload
%autoreload 2
```

```python id="I68Xx7ykRTnU"
import sys
sys.path.insert(0, './code')
```

```python id="N4hxE2mRRa-z"
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from scipy.sparse import csr_matrix
from implicit import als
from lightgbm import LGBMClassifier

from metrics import precision_at_k, recall_at_k
from utils import prefilter_items
from recommenders import MainRecommender

import warnings
warnings.filterwarnings('ignore')
```

```python id="cjOVeEKORgrb"
data = pd.read_parquet('./data/bronze/train.parquet.gzip')
item_features = pd.read_parquet('./data/bronze/products.parquet.gzip')
user_features = pd.read_parquet('./data/bronze/demographics.parquet.gzip')
```

```python id="MKIFLeq5Ru9Z"
ITEM_COL = 'item_id'
USER_COL = 'user_id'

# column processing
item_features.columns = [col.lower() for col in item_features.columns]
user_features.columns = [col.lower() for col in user_features.columns]

item_features.rename(columns={'product_id': ITEM_COL}, inplace=True)
user_features.rename(columns={'household_key': USER_COL }, inplace=True)
```

<!-- #region id="h6LdhM2PR0gP" -->
## Train-test split
<!-- #endregion -->

```python id="caCQa0TDR2OZ"
# Learning and validation scheme is important!
# - old purchases - | - 6 weeks - | - 3 weeks -
# select the size of the 2nd dataset (6 weeks) -> learning curve (dependence of the recall @ k metric on the size of the dataset)
VAL_MATCHER_WEEKS = 6
VAL_RANKER_WEEKS = 3

# take data for training matching model
data_train_matcher = data[data['week_no'] < data['week_no'].max() - (VAL_MATCHER_WEEKS + VAL_RANKER_WEEKS)]

# take data to validate the matching model
data_val_matcher = data[(data['week_no'] >= data['week_no'].max() - (VAL_MATCHER_WEEKS + VAL_RANKER_WEEKS)) &
                      (data['week_no'] < data['week_no'].max() - (VAL_RANKER_WEEKS))]

# take data for training ranking model
data_train_ranker = data_val_matcher.copy()  # For clarity. Next we will add changes and they will be different.

# take data for the test ranking, matching model
data_val_ranker = data[data['week_no'] >= data['week_no'].max() - VAL_RANKER_WEEKS]
```

```python id="oW8bKgHNSo3k"
def print_stats_data(df_data, name_df):
    print(name_df)
    print(f"Shape: {df_data.shape} Users: {df_data[USER_COL].nunique()} Items: {df_data[ITEM_COL].nunique()}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZzOHrKHRSmYx" outputId="f9ccb36a-2d33-454e-e872-b128538a5d71"
print_stats_data(data_train_matcher,'train_matcher')
print_stats_data(data_val_matcher,'val_matcher')
print_stats_data(data_train_ranker,'train_ranker')
print_stats_data(data_val_ranker,'val_ranker')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 131} id="YFtUzhYEStJx" outputId="7a199352-9760-4d03-b010-2be725988007"
data_train_matcher.head(2)
```

<!-- #region id="OL_iCq1DSufC" -->
## Pre-filter items
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="YN2OUnWUUCjF" outputId="1a493126-c64d-4faf-fd54-f18df3863d9d"
n_items_before = data_train_matcher['item_id'].nunique()

data_train_matcher = prefilter_items(data_train_matcher, item_features=item_features, take_n_popular=5000)

n_items_after = data_train_matcher['item_id'].nunique()
print('Decreased # items from {} to {}'.format(n_items_before, n_items_after))
```

<!-- #region id="-En_DI1OUjHj" -->
## Pre-filter users
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="-spyg-xiU9r1" outputId="9d65b1b0-6651-4fa4-aa11-d2ae3d9ead21"
# looking for common users
common_users = data_train_matcher.user_id.values

data_val_matcher = data_val_matcher[data_val_matcher.user_id.isin(common_users)]
data_train_ranker = data_train_ranker[data_train_ranker.user_id.isin(common_users)]
data_val_ranker = data_val_ranker[data_val_ranker.user_id.isin(common_users)]

print_stats_data(data_train_matcher,'train_matcher')
print_stats_data(data_val_matcher,'val_matcher')
print_stats_data(data_train_ranker,'train_ranker')
print_stats_data(data_val_ranker,'val_ranker')
```

<!-- #region id="fPt0KEO0Vcxo" -->
## Training candidate selection recommender
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 115, "referenced_widgets": ["f63c192c492c4ec79d85d88c9d0e40f0", "058769af67534e1fba8062859ffbed9d", "9a7b9776ba6645a58957b8aa0829cd92", "323fdb2416494ebf8d01213fd56249ad", "0e2686e8ad924be9a5da21776bbcce00", "6d36f4a086f647d599f690160b1c0fa9", "9babd00b1dc1442488a60ac271191c64", "bb1637a4b464470396445e6bdb2867c7", "78df5324bb544269897097e1094103ec", "bb9d0acc756542f38c1ac5a66ef14ecd", "70142dbc77b949cfaaef069386638f3b", "c7c3cec61f824648a54987ea71e50885", "c3627be3db20452aa58136d03423ed73", "c28d22a03f284218b5f8e7c6810f3703", "8e0c089e2d0c48679fbe8b5296652b8e", "d83b086094ed465baaebf438e66b119c"]} id="ZZR-FMxKVj3q" outputId="b37cbc59-347f-4bb8-975d-f7b8b251cdfb"
recommender = MainRecommender(data_train_matcher)
```

```python colab={"base_uri": "https://localhost:8080/"} id="14zRcVaoVtIx" outputId="49c33f05-bbc2-451c-824a-6cd8fd6eb33f"
recommender.get_als_recommendations(2375, N=5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="iSy28C_4WsdQ" outputId="b624b523-d7ea-4a9a-d737-e5821e6da2f9"
recommender.get_own_recommendations(2375, N=5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="-e-IZR_mWsYT" outputId="8c42494c-44ff-4366-8e6e-17b90f206d26"
recommender.get_similar_items_recommendation(2375, N=5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="yZPoeH5gWsUP" outputId="837f325e-c8af-4615-8f8c-874e4c5a14d9"
recommender.get_similar_users_recommendation(2375, N=5)
```

<!-- #region id="11J2aOl7Yh3u" -->
## Eval recall of matching
Measuring recall @ k. Quality is measured at data_val_matcher: next 6 weeks after the train
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 111} id="PUAX84OHYllQ" outputId="f7f044b3-8480-452e-b144-27b4c67c6f64"
ACTUAL_COL = 'actual'
result_eval_matcher = data_val_matcher.groupby(USER_COL)[ITEM_COL].unique().reset_index()
result_eval_matcher.columns=[USER_COL, ACTUAL_COL]
result_eval_matcher.head(2)
```

```python id="aZgczkOkW3ql"
# N = Neighbors
N_PREDICT = 50 

result_eval_matcher['own_rec'] = result_eval_matcher[USER_COL].apply(lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))
result_eval_matcher['sim_item_rec'] = result_eval_matcher[USER_COL].apply(lambda x: recommender.get_similar_items_recommendation(x, N=50))
result_eval_matcher['als_rec'] = result_eval_matcher[USER_COL].apply(lambda x: recommender.get_als_recommendations(x, N=50))
# result_eval_matcher['sim_user_rec'] = result_eval_matcher[USER_COL].apply(lambda x: recommender.get_similar_users_recommendation(x, N=50))
```

```python id="VdvbOQ7KY9v6"
def evalRecall(df_result, target_col_name, recommend_model):
    result_col_name = 'result'
    df_result[result_col_name] = df_result[target_col_name].apply(lambda x: recommend_model(x, N=25))
    return df_result.apply(lambda row: recall_at_k(row[result_col_name], row[ACTUAL_COL], k=N_PREDICT), axis=1).mean()
# evalRecall(result_eval_matcher, USER_COL, recommender.get_own_recommendations)

def calc_recall(df_data, top_k):
    for col_name in df_data.columns[2:]:
        yield col_name, df_data.apply(lambda row: recall_at_k(row[col_name], row[ACTUAL_COL], k=top_k), axis=1).mean()

def calc_precision(df_data, top_k):
    for col_name in df_data.columns[2:]:
        yield col_name, df_data.apply(lambda row: precision_at_k(row[col_name], row[ACTUAL_COL], k=top_k), axis=1).mean()
```

```python colab={"base_uri": "https://localhost:8080/"} id="G64ktburZB8B" outputId="764b45bb-3151-45b4-9492-a4f7e25f2cbb"
### Recall@50 of matching
TOPK_RECALL = 50
sorted(calc_recall(result_eval_matcher, TOPK_RECALL), key=lambda x: x[1],reverse=True)
[('own_rec', 0.06525657038145175),
 ('als_rec', 0.04739425945698195),
 ('sim_item_rec', 0.03399733646258331)]
```

```python colab={"base_uri": "https://localhost:8080/"} id="83KcnJtmZDRK" outputId="112b7119-4c50-44ed-a48f-c001edbbc7c6"
### Precision@5 of matching
TOPK_PRECISION = 5
sorted(calc_precision(result_eval_matcher, TOPK_PRECISION), key=lambda x: x[1],reverse=True)
```

<!-- #region id="6TpPQewCZL_l" -->
## Ranking
<!-- #endregion -->

<!-- #region id="W8rrwFKhdy4u" -->
```
# Train the Level 2 Model on Selected Candidates
# We train on data_train_ranking
# We train only on selected candidates
# For example, I will generate the top 50 candidates via get_own_recommendations
# Note: If the user has bought <50 products, then get_own_recommendations will add top-popular recommendations
# 3 time slots
# - old purchases - | - 6 weeks - | - 3 weeks -
```
<!-- #endregion -->

```python id="AXuGgLDAePAE"
# Preparing data for the train
# took users from the train for ranking
df_match_candidates = pd.DataFrame(data_train_ranker[USER_COL].unique())
df_match_candidates.columns = [USER_COL]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 111} id="mSWApvayeRUC" outputId="fe6db4fc-a6d3-4689-bbca-535f0b2a5fa1"
# collecting candidates from the first stage (matcher)
df_match_candidates['candidates'] = df_match_candidates[USER_COL].apply(lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))
df_match_candidates.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="aN6cngBNeg1m" outputId="15511a77-1bfe-431d-b900-24bda5616bdb"
df_items = df_match_candidates.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
df_items.name = 'item_id'
df_match_candidates = df_match_candidates.drop('candidates', axis=1).join(df_items)
df_match_candidates.head(4)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="h2dFoudIeq10" outputId="814ae50d-dc77-4cb6-90cf-33e556558643"
# Check warm start
print_stats_data(df_match_candidates, 'match_candidates')

# We create a training set for ranking taking into account candidates from stage 1
df_ranker_train = data_train_ranker[[USER_COL, ITEM_COL]].copy()
df_ranker_train ['target'] = 1 # there are only purchases
df_ranker_train.head ()
```

```python colab={"base_uri": "https://localhost:8080/"} id="atI-qxMge_0b" outputId="420a7887-02ab-4370-a1b3-049ecfd683b7"
# There are not enough zeros in the dataset, so we add our candidates to the quality of zeros
df_ranker_train = df_match_candidates.merge(df_ranker_train, on=[USER_COL, ITEM_COL], how='left')

# clean up duplicates
df_ranker_train = df_ranker_train.drop_duplicates(subset=[USER_COL, ITEM_COL])

df_ranker_train['target'].fillna(0, inplace = True)
df_ranker_train.target.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 111} id="ycNiTE_BdcLx" outputId="b8d35a74-296a-4273-c79e-e9fd44a57bff"
df_ranker_train.head(2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="KB6FHF7ffHLf" outputId="402606a0-d45e-4292-e4bd-3428e5970256"
df_ranker_train['target'].mean()
```

<!-- #region id="kY1Kt0f0fRoy" -->
<!-- #endregion -->

<!-- #region id="pEOuLNonfIWk" -->
For now, for ease of training, we will choose LightGBM with loss = binary. This is a classic binary classification

This is an example without feature generation
<!-- #endregion -->

<!-- #region id="QAUrXWv9iI4D" -->
Other good choices:
- classification via LightGBM
- CatBoost ranking via YetiRank, YetiRankPairwise
<!-- #endregion -->

<!-- #region id="nrvOq3OOfw0p" -->
### Preparing features for training the model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 128} id="f98iUM8LfxmP" outputId="aa0a1944-7fdf-4377-b7f2-1b1ad485490d"
item_features.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 165} id="IM1b8G2SfxkK" outputId="cc587c74-a71a-4f98-9caf-1e7315861a0a"
user_features.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 182} id="0mumVxewfxgd" outputId="5a3a9b21-5161-45b7-92d8-9f2ca88fd3d9"
df_ranker_train = df_ranker_train.merge(item_features, on='item_id', how='left')
df_ranker_train = df_ranker_train.merge(user_features, on='user_id', how='left')
df_ranker_train.head(2)
```

<!-- #region id="mmRFcz7Df8w4" -->
Features user_id:

- Average check
- Average purchase amount for 1 product in each category
- Number of purchases in each category
- Frequency of purchases once / month
- Share of shopping on weekends
- Share of purchases in the morning / afternoon / evening

Features item_id:

- Number of purchases per week
- Average number of purchases of 1 product per category per week
- (Number of purchases per week) / (Average number of purchases of 1 item per category per week)
- Price (Can be calculated from retil_train.csv)
- Price / Average price of a product in a category

Features of the pair user_id - item_id

- (Average purchase amount for 1 product in each category (we take the item_id category)) - (Price item_id)
- (Number of purchases by a user of a specific category per week) - (Average number of purchases by all users of a specific category per week)
- (Number of purchases by a user of a specific category per week) / (Average number of purchases by all users of a specific category per week)
<!-- #endregion -->

<!-- #region id="RtYP80IviSJ_" -->
**Note on possible features in the 2nd level model**
- Collaborative:
    - biases + embeddings from collaborative filtration / score item2item models
    - TF-IDF mattress items with> N purchases
    - TF-IDF + TSNE / UMAP
- handcrafted product features:
    - product categories
    - standardized frequency of purchase of goods for each customer
    - number of stores in which the product was sold
    - number of client transactions
    - mean / max / std number of unique items in the customer's cart
    - mean / max / std number of unique categories in the customer's cart
- handcrafted features for users:
    - Average check
    - Average price of one purchased item
    - Average number of days between purchases / since last purchase
    - Number of unique purchases across all categories transaction_id, product_id, store_id, level_i_id
    - Signs with accumulated bonuses
    - Average discount, share of purchased goods with discounts
- Interesting:
    - The fact of ordering each product in the last 5 transactions as a sequence of bits (categorical feature). 10001 - bought the item in the last transaction and 5 transactions ago (feature hashing)
    - item co_ocurrence
    - word2vec product embeddings (alternative name - item2vec, prod2vec)
    - Distance from word2vec product embedding to average user-bought product embeddings
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6HLBhvqLgEt8" outputId="98ea0771-141f-405a-c252-d4e1b7ee17a3"
X_train = df_ranker_train.drop('target', axis=1)
y_train = df_ranker_train[['target']]
cat_feats = X_train.columns[2:].tolist()
X_train[cat_feats] = X_train[cat_feats].astype('category')

cat_feats
```

<!-- #region id="-ZW8VFhVgNDs" -->
### Training the ranking model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 360} id="nMsNVazifcFB" outputId="1c44f3e3-8edb-4703-e28b-4e4ef1f7bbb6"
lgb = LGBMClassifier(objective='binary',
                     max_depth=8,
                     n_estimators=300,
                     learning_rate=0.05,
                     categorical_column=cat_feats)

lgb.fit(X_train, y_train)

train_preds = lgb.predict_proba(X_train)

df_ranker_predict = df_ranker_train.copy()
df_ranker_predict['proba_item_purchase'] = train_preds[:,1]
df_ranker_predict.head()
```

<!-- #region id="bXqSdY0ZgbB4" -->
We trained the ranking model on purchases from the data_train_ranker set and on candidates from own_recommendations, which is a training set, and now our task is to predict and evaluate on the test set.
<!-- #endregion -->

<!-- #region id="ArJcemDzgmXQ" -->
### Evaluation on test dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 111} id="IRJmxZaRglg7" outputId="2bd44384-7716-4348-ed13-ce9f5d1693b8"
result_eval_ranker = data_val_ranker.groupby(USER_COL)[ITEM_COL].unique().reset_index()
result_eval_ranker.columns=[USER_COL, ACTUAL_COL]
result_eval_ranker.head(2)
```

<!-- #region id="Afw2h48UgrOG" -->
### Eval matching on test dataset
<!-- #endregion -->

<!-- #region id="5px5q93Yg8kc" -->
measure precision only of the matching model to understand the impact of ranking on metrics
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xvy4lrlBgqU9" outputId="0ed383e7-64e7-41b7-bcba-aa502ce4efd3"
result_eval_ranker['own_rec'] = result_eval_ranker[USER_COL].apply(lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))
sorted(calc_precision(result_eval_ranker, TOPK_PRECISION), key=lambda x: x[1], reverse=True)
```

<!-- #region id="JPqcUlCsg0Or" -->
### Eval re-ranked matched result on test dataset
<!-- #endregion -->

<!-- #region id="7yEHmjW4g06l" -->
We take the top k predictions, ranked by probability, for each user
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JTTqyYalgbUL" outputId="3824eec7-128f-41b0-e07c-b27712216435"
def rerank(user_id):
    return df_ranker_predict[df_ranker_predict[USER_COL]==user_id].sort_values('proba_item_purchase', ascending=False).head(5).item_id.tolist()

result_eval_ranker['reranked_own_rec'] = result_eval_ranker[USER_COL].apply(lambda user_id: rerank(user_id))
print(*sorted(calc_precision(result_eval_ranker, TOPK_PRECISION), key=lambda x: x[1], reverse=True), sep='\n')
```
