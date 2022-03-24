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

<!-- #region id="g2TbyAJTABlF" -->
# Retail Product Recommendation with Negative Implicit Feedback
> A tutorial to demonstrate the process of training and evaluating various recommender models on a online retail store data. Along with the positive feedbacks like view, add-to-cart, we also have a negative event 'remove-from-cart'.

- toc: true
- badges: true
- comments: true
- categories: [retail]
- image: 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 836} executionInfo={"elapsed": 35130, "status": "ok", "timestamp": 1619353185930, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="GzunVZI1GxeQ" outputId="6f327d38-6eff-4181-cc78-a03d9c0c8c69"
#hide
!pip install git+https://github.com/maciejkula/spotlight.git@master#egg=spotlight
!git clone https://github.com/microsoft/recommenders.git
!pip install cornac
!pip install pandas==0.25.0
```

```python executionInfo={"elapsed": 6723, "status": "ok", "timestamp": 1619353203778, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="y-ofgNC3Y_RT"
#hide
import os
import sys
import math
import random
import datetime
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.model_selection import ParameterGrid

from fastai.collab import *
from fastai.tabular import *
from fastai.text import *

import cornac

from spotlight.interactions import Interactions
from spotlight.interactions import SequenceInteractions
from spotlight.cross_validation import random_train_test_split
from spotlight.cross_validation import user_based_train_test_split
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.evaluation import mrr_score
from spotlight.evaluation import precision_recall_score

from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from spotlight.cross_validation import user_based_train_test_split
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.evaluation import mrr_score
from spotlight.evaluation import precision_recall_score

from spotlight.interactions import SequenceInteractions
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.evaluation import sequence_mrr_score
from spotlight.evaluation import sequence_precision_recall_score

import warnings
warnings.filterwarnings("ignore")
```

```python executionInfo={"elapsed": 1307, "status": "ok", "timestamp": 1619353333233, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="U6E0vcJwMGDP"
#hide
sys.path.append('/content/recommenders/')
from reco_utils.dataset.python_splitters import python_chrono_split
from reco_utils.evaluation.python_evaluation import map_at_k
from reco_utils.evaluation.python_evaluation import precision_at_k
from reco_utils.evaluation.python_evaluation import ndcg_at_k 
from reco_utils.evaluation.python_evaluation import recall_at_k
from reco_utils.evaluation.python_evaluation import get_top_k_items
from reco_utils.recommender.cornac.cornac_utils import predict_ranking
```

<!-- #region id="5CynEys0q8s4" -->
## Data Loading
<!-- #endregion -->

```python executionInfo={"elapsed": 1523, "status": "ok", "timestamp": 1619353336991, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="vX1auAeuZB_K"
# loading data
df = pd.read_csv('rawdata.csv', header = 0,
                 names = ['event','userid','itemid','timestamp'],
                 dtype={0:'category', 1:'category', 2:'category'},
                 parse_dates=['timestamp'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1263, "status": "ok", "timestamp": 1619353338708, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="tZBH2b6VGX15" outputId="d426d3fa-352a-483d-d191-e4729530705b"
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1128, "status": "ok", "timestamp": 1619353338709, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="QTpzIHutLYeB" outputId="53607369-af1d-4023-97c2-c7bed938ab90"
df.info()
```

<!-- #region id="MwCDd4zLrBtM" -->
## Wrangling
<!-- #endregion -->

<!-- #region id="LhAPg2MwBRMJ" -->
### Removing Duplicates
<!-- #endregion -->

```python executionInfo={"elapsed": 1081, "status": "ok", "timestamp": 1619353341637, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="ELX_ANs0Fsfj"
# dropping exact duplicates
df = df.drop_duplicates()
```

<!-- #region id="nde6SCsdBTi1" -->
### Label Encoding
<!-- #endregion -->

```python executionInfo={"elapsed": 1488, "status": "ok", "timestamp": 1619353342200, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="XSfO7vQvFyEE"
# userid normalization
userid_encoder = preprocessing.LabelEncoder()
df.userid = userid_encoder.fit_transform(df.userid)

# itemid normalization
itemid_encoder = preprocessing.LabelEncoder()
df.itemid = itemid_encoder.fit_transform(df.itemid)
```

<!-- #region id="HnfFBs6Rrh8Z" -->
## Exploration
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 111} executionInfo={"elapsed": 721, "status": "ok", "timestamp": 1619353343821, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="4D8IVG7rMylq" outputId="51f604e3-f345-4c35-ce73-49c7253f424f"
df.describe().T
```

```python colab={"base_uri": "https://localhost:8080/", "height": 128} executionInfo={"elapsed": 848, "status": "ok", "timestamp": 1619353345855, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="RiRxYQbSM0mw" outputId="0339ae17-9d18-4fae-caa8-aaccd0cdfe94"
df.describe(exclude='int').T
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 726, "status": "ok", "timestamp": 1619353345856, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="DZDXdRukM2Yk" outputId="0aa5a6e8-1694-4555-ad72-8e72c06551d5"
df.timestamp.max() - df.timestamp.min()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1200, "status": "ok", "timestamp": 1619353346476, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="OpYyeSJULbgk" outputId="7ce2b5e1-fd64-448a-a355-97c167596ea0"
df.event.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1103, "status": "ok", "timestamp": 1619353348388, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="t9tXueqSM5ni" outputId="6e7fa21c-f17e-4ac8-9d46-325d803f1d32"
df.event.value_counts()/df.userid.nunique()
```

<!-- #region id="pwzFDW_IBNHV" -->
### User Interactions
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"elapsed": 3379, "status": "ok", "timestamp": 1619353350786, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="GrIHZLB8F1zj" outputId="4646251e-b4d9-4c49-ece6-22dd01c07869"
#hide-input
# User events
user_activity_count = dict()
for row in df.itertuples():
    if row.userid not in user_activity_count:
        user_activity_count[row.userid] = {'view_item':0, 
                                           'add_to_cart':0,
                                           'begin_checkout':0,
                                           'remove_from_cart':0, 
                                           'purchase':0}
    if row.event == 'view_item':
        user_activity_count[row.userid]['view_item'] += 1
    elif row.event == 'add_to_cart':
        user_activity_count[row.userid]['add_to_cart'] += 1
    elif row.event == 'begin_checkout':
        user_activity_count[row.userid]['begin_checkout'] += 1
    elif row.event == 'remove_from_cart':
        user_activity_count[row.userid]['remove_from_cart'] += 1
    elif row.event == 'purchase':
        user_activity_count[row.userid]['purchase'] += 1

user_activity = pd.DataFrame(user_activity_count)
user_activity = user_activity.transpose()
user_activity['activity'] = user_activity.sum(axis=1)

tempDF = pd.DataFrame(user_activity.activity.value_counts()).reset_index()
tempDF.columns = ['#Interactions','#Users']
sns.scatterplot(x='#Interactions', y='#Users', data=tempDF);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 280} executionInfo={"elapsed": 4384, "status": "ok", "timestamp": 1619353351926, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="s4oQQNgxM_bb" outputId="325ba409-2662-483d-bd38-4b07d95d0686"
#hide
df_activity = user_activity.copy()
event = df_activity.columns.astype('str')
sns.countplot(df_activity.loc[df_activity[event[0]]>0,event[0]]);
```

<!-- #region id="-kXFMUmABcL6" -->
### Add-to-cart Event Counts
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 280} executionInfo={"elapsed": 4871, "status": "ok", "timestamp": 1619353352563, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="G1YLJyrlNFSP" outputId="1195a5b4-a3dc-41d7-be2a-cd466ba6fa42"
#hide-input
sns.countplot(df_activity.loc[df_activity[event[1]]>0,event[1]])
plt.show()
```

<!-- #region id="phSQ9J6ZBe5l" -->
### Purchase Event Counts
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"elapsed": 1176, "status": "ok", "timestamp": 1619353355530, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="ghMH6rE2NLwD" outputId="f27d49b1-1f8a-427d-d195-94f097cd322e"
#hide-input
sns.countplot(df_activity.loc[df_activity[event[4]]>0,event[4]])
plt.show()
```

<!-- #region id="o-rNhn7XBkaw" -->
### Item Interactions
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"elapsed": 2434, "status": "ok", "timestamp": 1619353357303, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Oi9YOTovF8Gr" outputId="1ed5b722-bbc8-417a-d648-75a41bf5d9bf"
#hide-input
# item events
item_activity_count = dict()
for row in df.itertuples():
    if row.itemid not in item_activity_count:
        item_activity_count[row.itemid] = {'view_item':0, 
                                           'add_to_cart':0,
                                           'begin_checkout':0,
                                           'remove_from_cart':0, 
                                           'purchase':0}
    if row.event == 'view_item':
        item_activity_count[row.itemid]['view_item'] += 1
    elif row.event == 'add_to_cart':
        item_activity_count[row.itemid]['add_to_cart'] += 1
    elif row.event == 'begin_checkout':
        item_activity_count[row.itemid]['begin_checkout'] += 1
    elif row.event == 'remove_from_cart':
        item_activity_count[row.itemid]['remove_from_cart'] += 1
    elif row.event == 'purchase':
        item_activity_count[row.itemid]['purchase'] += 1

item_activity = pd.DataFrame(item_activity_count)
item_activity = item_activity.transpose()
item_activity['activity'] = item_activity.sum(axis=1)

tempDF = pd.DataFrame(item_activity.activity.value_counts()).reset_index()
tempDF.columns = ['#Interactions','#Items']
sns.scatterplot(x='#Interactions', y='#Items', data=tempDF);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 260} executionInfo={"elapsed": 1700, "status": "ok", "timestamp": 1619353359253, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="_iAX_77jNPVx" outputId="1555017f-840f-43ab-b20d-e401dc3a213d"
#hide
plt.rcParams['figure.figsize'] = 15,3
data = pd.DataFrame(pd.to_datetime(df['timestamp'], infer_datetime_format=True))
data['Count'] = 1
data.set_index('timestamp', inplace=True)
data = data.resample('D').apply({'Count':'count'})
ax = data['Count'].plot(marker='o', linestyle='-')
```

<!-- #region id="hLc0MNZ_88IC" -->
## Rule-based Approaches
<!-- #endregion -->

<!-- #region id="1GiTyrgba3Y3" -->
### Top-N Trending Products
<!-- #endregion -->

```python executionInfo={"elapsed": 1080, "status": "ok", "timestamp": 1619353363012, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="YAcqM0OemhTx"
#collapse
def top_trending(n, timeperiod, timestamp):
  start = str(timestamp.replace(microsecond=0) - pd.Timedelta(minutes=timeperiod))
  end = str(timestamp.replace(microsecond=0))
  trending_items = df.loc[(df.timestamp.between(start,end) & (df.event=='view_item')),:].sort_values('timestamp', ascending=False)
  return trending_items.itemid.value_counts().index[:n]
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 925, "status": "ok", "timestamp": 1619353428637, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="1VtxQn-4wb9X" outputId="2adba715-3bb3-4a24-ed4b-60a75aac00c2"
user_current_time = df.timestamp[100]
top_trending(5, 50, user_current_time)
```

<!-- #region id="WIk9-LVI80xF" -->
### Top-N Least Viewed Items
<!-- #endregion -->

```python executionInfo={"elapsed": 1149, "status": "ok", "timestamp": 1619353444483, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="He-d6z4woWQY"
#collapse
def least_n_items(n=10):
  temp1 = df.loc[df.event=='view_item'].groupby(['itemid'])['event'].count().sort_values(ascending=True).reset_index()
  temp2 = df.groupby('itemid').timestamp.max().reset_index()
  item_ids = pd.merge(temp1,temp2,on='itemid').sort_values(['event', 'timestamp'], ascending=[True, False]).reset_index().loc[:n-1,'itemid']
  return itemid_encoder.inverse_transform(item_ids.values)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 972, "status": "ok", "timestamp": 1619353444485, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="RWC7mpYHrZ4G" outputId="52696f3d-b26b-4c51-fc7d-86f05be0f020"
least_n_items(10)
```

<!-- #region id="-othcGTQGgO1" -->
# Data Transformation
<!-- #endregion -->

<!-- #region id="v3rVZGmXGj6V" -->
Many times there are no explicit ratings or preferences given by users, that is, the interactions are usually implicit. This
information may reflect users' preference towards the items in an implicit manner.

Option 1 - Simple Count: The most simple technique is to count times of interactions between user and item for
producing affinity scores.

Option 2 - Weighted Count: It is useful to consider the types of different interactions as weights in the count
aggregation. For example, assuming weights of the three differen types, "click", "add", and "purchase", are 1, 2, and 3, respectively.

Option 3 - Time-dependent Count: In many scenarios, time dependency plays a critical role in preparing dataset for
building a collaborative filtering model that captures user interests drift over time. One of the common techniques for
achieving time dependent count is to add a time decay factor in the counting.

<!-- #endregion -->

<!-- #region id="46SxQnYfGpB6" -->
### A. Count
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1465, "status": "ok", "timestamp": 1619353450292, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="AKkgVA51EHZt" outputId="564ed79e-0094-49e7-b20a-580966666cce"
#collapse
data_count = df.groupby(['userid', 'itemid']).agg({'timestamp': 'count'}).reset_index()
data_count.columns = ['userid', 'itemid', 'affinity']
data_count.head()
```

<!-- #region id="ZTAVEqEsGujg" -->
### B. Weighted Count
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1023, "status": "ok", "timestamp": 1619353453966, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Vch5felgHsk6" outputId="4e8809a8-50f2-4b61-e3b7-b2c1f1cfab24"
#hide
data_w = df.loc[df.event!='remove_from_cart',:]

affinity_weights = {
    'view_item': 1,
    'add_to_cart': 3, 
    'begin_checkout': 5, 
    'purchase': 6,
    'remove_from_cart': 3
}

data_w['event'].apply(lambda x: affinity_weights[x])

data_w.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1375, "status": "ok", "timestamp": 1619353454594, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="6lT4A3uBGy0c" outputId="520d3359-dc6d-4b4c-f7bd-5d8902203cf6"
#collapse
data_w['weight'] = data_w['event'].apply(lambda x: affinity_weights[x])
data_wcount = data_w.groupby(['userid', 'itemid'])['weight'].sum().reset_index()
data_wcount.columns = ['userid', 'itemid', 'affinity']
data_wcount.head()
```

<!-- #region id="2JRTP72-I9a-" -->
### C. Time dependent Count
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 16314, "status": "ok", "timestamp": 1619353472907, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Gudy30cdLozz" outputId="f7afdf22-6273-48ed-9b50-15c08f5cdccf"
#hide
T = 30
t_ref = datetime.datetime.utcnow()

data_w['timedecay'] = data_w.apply(
    lambda x: x['weight'] * math.exp(-math.log2((t_ref - pd.to_datetime(x['timestamp']).tz_convert(None)).days / T)), 
    axis=1
)

data_w.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 16171, "status": "ok", "timestamp": 1619353472908, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="S2TaBSliLoyg" outputId="bef02246-a4c4-4b71-da91-0e9e96464587"
#collapse
data_wt = data_w.groupby(['userid', 'itemid'])['timedecay'].sum().reset_index()
data_wt.columns = ['userid', 'itemid', 'affinity']
data_wt.head()
```

<!-- #region id="adGKJl-lL2Xc" -->
# Train Test Split
<!-- #endregion -->

<!-- #region id="L07a4lZhGwek" -->
Option 1 - Random Split: Random split simply takes in a data set and outputs the splits of the data, given the split
ratios

Option 2 - Chronological split: Chronogically splitting method takes in a dataset and splits it on timestamp

<!-- #endregion -->

```python executionInfo={"elapsed": 1116, "status": "ok", "timestamp": 1619353479518, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="jscWQX-GL2M0"
#collapse
data = data_w[['userid','itemid','timedecay','timestamp']]

col = {
  'col_user': 'userid',
  'col_item': 'itemid',
  'col_rating': 'timedecay',
  'col_timestamp': 'timestamp',
}

col3 = {
  'col_user': 'userid',
  'col_item': 'itemid',
  'col_timestamp': 'timestamp',
}

train, test = python_chrono_split(data, ratio=0.75, min_rating=10, 
                                  filter_by='user', **col3)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"elapsed": 891, "status": "ok", "timestamp": 1619353508985, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="MEZG8CgOL7zT" outputId="dd001a8d-d11c-45d7-b1bc-23758dda5ed6"
train.loc[train.userid==7,:]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 111} executionInfo={"elapsed": 926, "status": "ok", "timestamp": 1619353511644, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="wLUBNsN9L7uj" outputId="846ca9bd-1396-4174-ba23-2d672fcb4488"
test.loc[test.userid==7,:]
```

<!-- #region id="rC3QDUu9DbcD" -->
# Experiments
<!-- #endregion -->

<!-- #region id="cOnCVSh3MK4N" -->
### Item Popularity Recomendation Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 938, "status": "ok", "timestamp": 1619353515152, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="xRfxWKcsMKtE" outputId="fcda236e-2a14-49e4-a169-be0565743e8e"
#hide
# Recommending the most popular items is intuitive and simple approach
item_counts = train['itemid'].value_counts().to_frame().reset_index()
item_counts.columns = ['itemid', 'count']
item_counts.head()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11809, "status": "ok", "timestamp": 1619353528281, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="0tZVGvlyMKqQ" outputId="0d7e8025-cf23-4304-f8cd-2a19d415eabf"
#hide
user_item_col = ['userid', 'itemid']

# Cross join users and items
test_users = test['userid'].unique()
user_item_list = list(itertools.product(test_users, item_counts['itemid']))
users_items = pd.DataFrame(user_item_list, columns=user_item_col)

print("Number of user-item pairs:", len(users_items))

# Remove seen items (items in the train set) as we will not recommend those again to the users
from reco_utils.dataset.pandas_df_utils import filter_by
users_items_remove_seen = filter_by(users_items, train, user_item_col)

print("After remove seen items:", len(users_items_remove_seen))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 11645, "status": "ok", "timestamp": 1619353528283, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="6rz-PTnDMKn_" outputId="de202645-1a9c-4ac6-b2d6-6709c8085bb6"
# Generate recommendations
baseline_recommendations = pd.merge(item_counts, users_items_remove_seen, 
                                    on=['itemid'], how='inner')
baseline_recommendations.head()
```

```python id="hxY9uI_-DAZw"
#hide
k = 10

cols = {
  'col_user': 'userid',
  'col_item': 'itemid',
  'col_rating': 'timedecay',
  'col_prediction': 'count',
}

eval_map = map_at_k(test, baseline_recommendations, k=k, **cols)
eval_ndcg = ndcg_at_k(test, baseline_recommendations, k=k, **cols)
eval_precision = precision_at_k(test, baseline_recommendations, k=k, **cols)
eval_recall = recall_at_k(test, baseline_recommendations, k=k, **cols)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 38152, "status": "ok", "timestamp": 1619353554935, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="ExmNutV8MKjv" outputId="e33ca3d2-9634-4459-a9ca-c86a097e4b89"
print("MAP:\t%f" % eval_map,
      "NDCG@K:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
```

<!-- #region id="b_c-SVPpN4m4" -->
### Cornac BPR Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 100, "referenced_widgets": ["eac0651563a24e698eb8203b1f8e4b68", "1b3df8a975e8479690d9b23df9db5271", "f7bfd6ea4bfb4d1ca7e846ec8db49fb9", "9a928c20fe8a474a870a42f92995f7a7", "d3706039787a45d9a4005e8ca1fa9dbd", "925673dfcae74e57bb8bba183e1442fb", "ac36e1c7b8c54cc1bb8e2dbf5642b745", "c4bd4f149a00418ab9321a8fc76aa4b8"]} executionInfo={"elapsed": 7221, "status": "ok", "timestamp": 1619353557367, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="IuTv44I3N7CH" outputId="96270b1c-d4eb-42e3-8231-d48ebfde8471"
#hide
TOP_K = 10
NUM_FACTORS = 200
NUM_EPOCHS = 100
SEED = 40

train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

bpr = cornac.models.BPR(
    k=NUM_FACTORS,
    max_iter=NUM_EPOCHS,
    learning_rate=0.01,
    lambda_reg=0.001,
    verbose=True,
    seed=SEED
)

from reco_utils.common.timer import Timer
with Timer() as t:
    bpr.fit(train_set)
print("Took {} seconds for training.".format(t))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11557, "status": "ok", "timestamp": 1619353562353, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="7cdbKp0IOc20" outputId="03480d61-c869-4f55-a0e9-cac80781e891"
#hide
with Timer() as t:
    all_predictions = predict_ranking(bpr, train, usercol='userid', itemcol='itemid', remove_seen=True)
print("Took {} seconds for prediction.".format(t))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 6263, "status": "ok", "timestamp": 1619353562354, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="lxZvFv4EOc0U" outputId="d0f22607-801f-4b78-fb33-1b5745d996fc"
all_predictions.head()
```

```python id="hhano5sPDXSP"
#hide
k = 10
cols = {
  'col_user': 'userid',
  'col_item': 'itemid',
  'col_rating': 'timedecay',
  'col_prediction': 'prediction',
}

eval_map = map_at_k(test, all_predictions, k=k, **cols)
eval_ndcg = ndcg_at_k(test, all_predictions, k=k, **cols)
eval_precision = precision_at_k(test, all_predictions, k=k, **cols)
eval_recall = recall_at_k(test, all_predictions, k=k, **cols)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 49857, "status": "ok", "timestamp": 1619351355802, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="7ISv6qwuOe-S" outputId="226aef91-bfaa-4790-8834-2c5f7fdb7cc8"
#hide-input
print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
```

<!-- #region id="b2B-pTkNO2sW" -->
### SARS Model
<!-- #endregion -->

```python id="_WoGMBppO5YR"
#collapse
from reco_utils.recommender.sar.sar_singlenode import SARSingleNode

TOP_K = 10

header = {
    "col_user": "userid",
    "col_item": "itemid",
    "col_rating": "timedecay",
    "col_timestamp": "timestamp",
    "col_prediction": "prediction",
}

model = SARSingleNode(
    similarity_type="jaccard", 
    time_decay_coefficient=0, 
    time_now=None, 
    timedecay_formula=False, 
    **header
)

model.fit(train)
```

```python id="mGM9VxzWDfLJ"
#hide
top_k = model.recommend_k_items(test, remove_seen=True)

# all ranking metrics have the same arguments
args = [test, top_k]
kwargs = dict(col_user='userid', 
              col_item='itemid', 
              col_rating='timedecay', 
              col_prediction='prediction', 
              relevancy_method='top_k', 
              k=TOP_K)

eval_map = map_at_k(*args, **kwargs)
eval_ndcg = ndcg_at_k(*args, **kwargs)
eval_precision = precision_at_k(*args, **kwargs)
eval_recall = recall_at_k(*args, **kwargs)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12937, "status": "ok", "timestamp": 1619351554207, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="gE3YREbHO5Wq" outputId="f72a9300-fa72-4f0c-98e9-6ed71debb321"
#hide-input
print(f"Model:",
      f"Top K:\t\t {TOP_K}",
      f"MAP:\t\t {eval_map:f}",
      f"NDCG:\t\t {eval_ndcg:f}",
      f"Precision@K:\t {eval_precision:f}",
      f"Recall@K:\t {eval_recall:f}", sep='\n')
```

<!-- #region id="Bcq2L_Wyrm88" -->
### Spotlight

<!-- #endregion -->

<!-- #region id="O4KcrGvO62zn" -->
#### Implicit Factorization Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 53336, "status": "ok", "timestamp": 1619350189787, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Qse65DPbGRWg" outputId="f2ed98a9-f6fa-471c-8517-7c091a4c70bb"
#collapse
interactions = Interactions(user_ids = df.userid.astype('int32').values,
                            item_ids = df.itemid.astype('int32').values,
                            timestamps = df.timestamp.astype('int32'),
                            num_users = df.userid.nunique(),
                            num_items = df.itemid.nunique())

train_user, test_user = random_train_test_split(interactions, test_percentage=0.2)

model = ImplicitFactorizationModel(loss='bpr', embedding_dim=64, n_iter=10, 
                                   batch_size=256, l2=0.0, learning_rate=0.01, 
                                   optimizer_func=None, use_cuda=False, 
                                   representation=None, sparse=False, 
                                   num_negative_samples=10)

model.fit(train_user, verbose=1)

pr = precision_recall_score(model, test=test_user, train=train_user, k=10)
print('Pricison@10 is {:.3f} and Recall@10 is {:.3f}'.format(pr[0].mean(), pr[1].mean()))
```

<!-- #region id="M45zFAXJ6vob" -->
Implicit Factorization Model with Grid Search
<!-- #endregion -->

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="aA_fzKCZJ7W_" outputId="b4b23257-c7d8-44a4-89af-c9a75e508a81"
#hide
interactions = Interactions(user_ids = df.userid.astype('int32').values,
                            item_ids = df.itemid.astype('int32').values,
                            timestamps = df.timestamp.astype('int32'),
                            num_users = df.userid.nunique(),
                            num_items = df.itemid.nunique())

train_user, test_user = random_train_test_split(interactions, test_percentage=0.2)

params_grid = {'loss':['bpr', 'hinge'],
               'embedding_dim':[32, 64],
               'learning_rate': [0.01, 0.05, 0.1],
               'num_negative_samples': [5,10,50]
               }
grid = ParameterGrid(params_grid)

for p in grid:
  model = ImplicitFactorizationModel(**p, n_iter=10, batch_size=256, l2=0.0, 
                                    optimizer_func=None, use_cuda=False, 
                                    representation=None, sparse=False)
  model.fit(train_user, verbose=1)
  pr = precision_recall_score(model, test=test_user, train=train_user, k=10)
  print('Pricison@10 is {:.3f} and Recall@10 is {:.3f}'.format(pr[0].mean(), pr[1].mean()))
```

<!-- #region id="s36VzY7e69n6" -->
#### CNN Pooling Sequence Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 21190, "status": "ok", "timestamp": 1619350230775, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="dUFd1yMwKm4j" outputId="ad8cdf25-da3f-4371-ea81-417bd1de7709"
#collapse
interactions = Interactions(user_ids = df.userid.astype('int32').values,
                            item_ids = df.itemid.astype('int32').values+1,
                            timestamps = df.timestamp.astype('int32'))

train, test = random_train_test_split(interactions, test_percentage=0.2)
train_seq = train.to_sequence(max_sequence_length=10)
test_seq = test.to_sequence(max_sequence_length=10)

model = ImplicitSequenceModel(loss='bpr', representation='pooling', 
                              embedding_dim=32, n_iter=10, batch_size=256, 
                              l2=0.0, learning_rate=0.01, optimizer_func=None, 
                              use_cuda=False, sparse=False, num_negative_samples=5)

model.fit(train_seq, verbose=1)

mrr_seq = sequence_mrr_score(model, test_seq)
mrr_seq.mean()
```

<!-- #region id="Aoxq92grr8sd" -->
## FastAI CollabLearner
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1179, "status": "ok", "timestamp": 1619350280860, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Q5uwIBUMLt8x" outputId="54fe269c-a73c-4270-b529-c1f298139291"
#hide
df['rating'] = df['event'].map({'view_item': 1,
                                'add_to_cart': 2,
                                'begin_checkout': 3, 
                                'purchase': 5,
                               'remove_from_cart': 0,
                                })

ratings = df[["userid", 'itemid', "rating", 'timestamp']].copy()

data = CollabDataBunch.from_df(ratings, seed=42)
data
```

```python colab={"base_uri": "https://localhost:8080/", "height": 320} executionInfo={"elapsed": 4123, "status": "ok", "timestamp": 1619350292377, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="03eIhC1aL-eq" outputId="3d9ad68d-9a16-4eb4-e30b-02229ff59956"
#hide
learn = collab_learner(data, n_factors=50, y_range=[0,5.5])
learn.lr_find()
learn.recorder.plot(skip_end=15)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 80} executionInfo={"elapsed": 21044, "status": "ok", "timestamp": 1619350312768, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="h5ssccv9MEOh" outputId="8fb7eaa9-3636-48f6-89bb-9a6b7f2934da"
learn.fit_one_cycle(1, 5e-6)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} executionInfo={"elapsed": 16533, "status": "ok", "timestamp": 1619350313136, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="1Purf6QTMGGK" outputId="9a3bb223-45d2-4c94-c40b-6c5ec1ee659f"
learn.summary()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} executionInfo={"elapsed": 182271, "status": "ok", "timestamp": 1619350581109, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="mGtcXKJqMF54" outputId="93070b70-886a-4ba3-ecaa-43263ddec9e2"
learn.fit(10, 1e-3)
```
