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

# Itempop and two-stage recommender on MTS data

<!-- #region id="Ey05k9RtFXlQ" -->
## Setup
<!-- #endregion -->

```python id="OZHUcZZCmCxf"
!pip install --upgrade pip setuptools wheel
!git clone https://github.com/benfred/implicit
!cd implicit && pip install .
!pip install -q catboost
!pip install recohut
```

```python id="EJ_UIHjq9NnK"
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

import random
import datetime

import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from implicit import nearest_neighbours as NN
from implicit.nearest_neighbours import TFIDFRecommender

from catboost import CatBoostClassifier

from recohut.datasets.mts import MTSDataset
from recohut.utils.common_utils import get_coo_matrix
from recohut.transforms.splitting import TimeRangeSplit
from recohut.models.itempop import ItemPop as PopularRecommender
```

```python id="nmhnVyko6Ynx"
ds = MTSDataset(data_dir='/content/data', sample_frac=0.1)
```

```python id="UJwOSLrd9Tmg"
users_df = pd.read_csv(os.path.join(ds.processed_dir, 'users_processed.csv'))
items_df = pd.read_csv(os.path.join(ds.processed_dir, 'items_processed.csv'))
interactions_df = pd.read_csv(os.path.join(ds.processed_dir, 'interactions_processed.csv'))
```

```python id="lQiBlArS-IVP"
interactions_df['last_watch_dt'] = pd.to_datetime(interactions_df['last_watch_dt'])
interactions_df.sort_values(by='last_watch_dt', inplace=True)
```

<!-- #region id="NJhm264qm293" -->
## Winning Solution

This solution includes a two-stage model. I used item-item CF from implicit library to generate candidates with their scores and Catboost classifier to predict final ranks with classification objective. Recommendations for cold users were made with Popular items.

Implicit model parameters were chosen on sliding time window cross validation. The best scores were achieved by Cosine recommender model, taking only last 20 interactions for each user. 100 candidates with their scores were generated for each user, filtering all items that user had interactions with.

Implicit candidates were calculated for the last 14 days of the interactions. Then catboost model was trained on positive interactions from the candidates list on last 14 days. Random negative sampling was applied.

For final submission implicit candidates and catboost predictions were recalculated on the whole dataset.

Ref: [Daria](https://github.com/blondered/ods_MTS_RecSys_Challenge_solution)
<!-- #endregion -->

```python id="Oa4FR6zv_lsB"
# Creating items and users mapping
users_inv_mapping = dict(enumerate(interactions_df['user_id'].unique()))
users_mapping = {v: k for k, v in users_inv_mapping.items()}
items_inv_mapping = dict(enumerate(interactions_df['item_id'].unique()))
items_mapping = {v: k for k, v in items_inv_mapping.items()}
```

```python id="FbsUVNeImJrm"
# Preparing data
last_date_df = interactions_df['last_watch_dt'].max()
boosting_split_date = last_date_df - pd.Timedelta(days=14)
boosting_data = interactions_df[(interactions_df['last_watch_dt'] >
                                 boosting_split_date)].copy()
boost_idx = boosting_data['user_id'].unique() 
before_boosting = interactions_df[(interactions_df['last_watch_dt'] <=
                                   boosting_split_date)].copy()
before_boosting_known_items = before_boosting.groupby(
    'user_id')['item_id'].apply(list).to_dict()

before_boosting_known_items_mapped = {}
for user, recommend in before_boosting_known_items.items():
    before_boosting_known_items_mapped[user] = list(map(lambda x:
                                                        items_mapping[x],
                                                        recommend))
before_boosting['order_from_recent'] = before_boosting.sort_values(
    by=['last_watch_dt'], ascending=False).groupby('user_id').cumcount() + 1
boost_warm_idx = np.intersect1d(before_boosting['user_id'].unique(),
                                boosting_data['user_id'].unique())
```

<!-- #region id="70FwEuTwIYD4" -->
 Calculates top candidates from implicit model with their scores. Implicit parameters were chosen on time range split cross-validation. History offset stands for taking only lask X items from user history. Day offset stands for taking items from last X days of user history.
<!-- #endregion -->

```python id="9X0MRF9TBLvs"
k_neighbours = 200
day_offset = 170
history_offset = 20
distance = 'Cosine'
num_candidates = 100
```

```python id="-S2vvbAWBTAZ"
before_boosting['order_from_recent'] = before_boosting.sort_values(
    by=['last_watch_dt'], ascending=False).groupby('user_id').cumcount() + 1
train = before_boosting.copy()
date_window = train['last_watch_dt'].max() - pd.DateOffset(days=day_offset)
train = train[train['last_watch_dt'] >= date_window]
```

```python id="b1sORvcLCc2V"
if history_offset:
    train = train[train['order_from_recent'] < history_offset]
    
if distance == 'Cosine':
    model = NN.CosineRecommender(K=k_neighbours)
    weights = None
else:
    model = NN.TFIDFRecommender(K=k_neighbours)
    weights = None
```

```python id="XL39gZ51CnWc"
train_mat = get_coo_matrix(
    train,
    users_mapping=users_mapping,
    items_mapping=items_mapping,
    weight_col=weights
).tocsr()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["1150ae9b77d04ee089151cdf9b3c97fd", "6404fcf9360b4c8bafac3d0c62dc7d58", "cd203aa954364d05a2d9197f04bac18a", "7d5a993575214d4189c013bb12fc9080", "552747c596b440929459610765a70c67", "6b9c40da36c746169708e1251c893b47", "e6fb3757a3db480aa1be1f0a91e19f4d", "163ead0dd46c434ea3412e462a0938db", "b6183320801a4148b75f6b36b65a9b13", "6773a61987794d45bf453ac8a8f78a34", "1cfcd7deb21741cc95481b1ece102ced"]} executionInfo={"elapsed": 29070, "status": "ok", "timestamp": 1642187764981, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="sjI4cw1ZCpoz" outputId="6660fc68-d9fa-4477-8a4a-10b9073d4b0f"
model.fit(train_mat.T, show_progress=True)
```

```python id="gnnc6uFECxM5"
def generate_implicit_recs_mapper(
        model,
        train_matrix,
        top_N,
        user_mapping,
        item_inv_mapping,
        filter_already_liked_items,
        known_items=None,
        filter_items=None,
        return_scores=False
):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        if filter_items:
            if user in known_items:
                filtering = set(known_items[user]).union(set(filter_items))
            else:
                filtering = filter_items
        else:
            if known_items and user in known_items:
                filtering = known_items[user]
            else:
                filtering = None
        recs = model.recommend(user_id,
                               train_matrix,
                               N=top_N,
                               filter_already_liked_items=filter_already_liked_items,
                               filter_items=filtering)
        if return_scores:
            return recs
        return recs[0]

    return _recs_mapper
```

```python id="GhInkhX_C17s"
mapper = generate_implicit_recs_mapper(
    model,
    train_mat,
    num_candidates,
    users_mapping,
    items_inv_mapping,
    filter_already_liked_items=False,
    known_items=before_boosting_known_items_mapped,
    filter_items=None,
    return_scores=True
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} executionInfo={"elapsed": 6395, "status": "ok", "timestamp": 1642188036867, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="pWPnNFW-4phP" outputId="0b6f43f5-0329-4622-fec5-82d438260602"
recs = pd.DataFrame({'user_id': boost_warm_idx})
recs['item_id_score'] = recs['user_id'].map(mapper)
recs['item_id'] = recs['item_id_score'].apply(lambda x: x[0])
recs['implicit_score'] = recs['item_id_score'].apply(lambda x: x[1])
recs['tmp'] = recs.apply(lambda row: list(zip(row['item_id'], row['implicit_score'])), axis=1) 
recs = recs.explode('tmp')
recs[['item_id','implicit_score']] = pd.DataFrame(recs['tmp'].tolist(), index=recs.index)
recs.drop(columns='tmp', inplace=True)
recs.drop(['item_id_score'], axis=1, inplace=True)
recs
```

```python id="roQ4due5DKVl"
recs.to_csv(os.path.join(ds.processed_dir, 'impl_scores.csv'), index=False)
```

```python id="Xje9T6CtI7kM"
# taking candidates from implicit model and generating positive samples
candidates = pd.read_csv(os.path.join(ds.processed_dir, 'impl_scores.csv'))
candidates['item_id'] = candidates['item_id'].fillna(0.).astype('int64')
candidates['id'] = candidates.index
pos = candidates.merge(boosting_data[['user_id', 'item_id']], 
                       on=['user_id', 'item_id'], how='inner')
pos['target'] = 1
```

```python colab={"base_uri": "https://localhost:8080/", "height": 677} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1642188049285, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="XmD3UFnzK4el" outputId="b11e5bae-1d3b-41f9-af99-6a0c4c532065"
pos
```

```python id="xnuzDSyeKhEa"
# Generating negative samples
num_negatives = 3
pos_group = pos.groupby('user_id')['item_id'].count()
neg = candidates[~candidates['id'].isin(pos['id'])].copy()
neg_sampling = pd.DataFrame(neg.groupby('user_id')['id'].apply(
    list)).join(pos_group, on='user_id',  rsuffix='p', how='right')
neg_sampling['num_choices'] = np.clip(neg_sampling['item_id'] * num_negatives, 
                                      a_min=0, a_max=25)
func = lambda row: np.random.choice(row['id'],
                                    size=row['num_choices'],
                                    replace=False)
neg_sampling['sample_idx'] = neg_sampling.apply(func, axis=1)
idx_chosen = neg_sampling['sample_idx'].explode().values
neg = neg[neg['id'].isin(idx_chosen)]
neg['target'] = 0
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1642188051315, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="iReJGyJNMpVg" outputId="adcff675-6618-47f3-a534-67ef145b3ccb"
neg
```

```python id="B3h45yA7MzMe"
# Creating training data sample and early stopping data sample
boost_idx_train = np.intersect1d(boost_idx, pos['user_id'].unique())
boost_train_users, boost_eval_users = train_test_split(boost_idx_train, 
                                                       test_size=0.1,
                                                       random_state=345)
select_col = ['user_id', 'item_id', 'implicit_score', 'target']
boost_train = shuffle(
    pd.concat([
               pos[pos['user_id'].isin(boost_train_users)],
               neg[neg['user_id'].isin(boost_train_users)]
    ])[select_col]
)
boost_eval = shuffle(
    pd.concat([
               pos[pos['user_id'].isin(boost_eval_users)],
               neg[neg['user_id'].isin(boost_eval_users)]
    ])[select_col]
)
```

```python id="ojeLRc9iM-LQ"
user_col = ['user_id','age','income','sex','kids_flg','boost_user_watch_cnt_all',
            'boost_user_watch_cnt_last_14']

item_col = ['item_id','content_type','countries_max','for_kids','age_rating',
            'studios_max','genres_max','genres_min','genres_med','release_novelty']

item_stats_col = ['item_id','watched_in_7_days','watch_ts_std','trend_slope',
                  'watch_ts_quantile_95_diff','watch_ts_median_diff',
                  'watched_in_all_time','male_watchers_fraction',
                  'female_watchers_fraction','younger_35_fraction','older_35_fraction']
                  
cat_col = ['age','income','sex','content_type']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 364} executionInfo={"elapsed": 610, "status": "ok", "timestamp": 1642188056980, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="-gP7TfKGNMba" outputId="e9484e51-24c6-4da8-e152-b51c1c68bc9e"
train_feat = boost_train.merge(users_df[user_col],
                               on=['user_id'],
                               how='left')\
                               .merge(items_df[item_col],
                                      on=['item_id'],
                                      how='left')
                               
eval_feat = boost_eval.merge(users_df[user_col],
                             on=['user_id'],
                             how='left') \
                               .merge(items_df[item_col],
                                      on=['item_id'],
                                      how='left')
                               
eval_feat
```

```python colab={"base_uri": "https://localhost:8080/", "height": 488} executionInfo={"elapsed": 623, "status": "ok", "timestamp": 1642188058306, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="ECh2RhahNSsQ" outputId="2cafea2d-7165-47ce-a45b-cc5acd6840e6"
item_stats = pd.read_csv(os.path.join(ds.processed_dir, 'item_stats.csv'))
item_stats = item_stats[item_stats_col]
train_feat = train_feat.join(item_stats.set_index('item_id'), 
                             on='item_id', how='left')
eval_feat = eval_feat.join(item_stats.set_index('item_id'), 
                           on='item_id', how='left')
drop_col = ['user_id', 'item_id']
target_col = ['target']

X_train = train_feat.drop(drop_col + target_col, axis=1)
y_train = train_feat[target_col]
X_val = eval_feat.drop(drop_col + target_col, axis=1)
y_val = eval_feat[target_col]
X_train.fillna('None', inplace=True)
X_val.fillna('None', inplace=True)
X_train[cat_col] = X_train[cat_col].astype('category')
X_val[cat_col] = X_val[cat_col].astype('category')

X_train
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3256, "status": "ok", "timestamp": 1642188064223, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="NPARxAndNoNV" outputId="7d21dbf9-a091-40c6-990e-84db9fb7f9a1"
# Training CatBoost classifier with parameters previously chosen on cross validation
params = {
    'subsample': 0.97, 
    'max_depth': 9,
    'n_estimators': 2000,
    'learning_rate': 0.03, 
    'scale_pos_weight': num_negatives, 
    'l2_leaf_reg': 27, 
    'thread_count': -1,
    'verbose': 200,
    'task_type': "CPU",
    'devices': '0:1',
    # 'bootstrap_type': 'Poisson'
}
boost_model = CatBoostClassifier(**params)
boost_model.fit(X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=200,
                cat_features=cat_col,
                plot=False)
```

```python id="KH3ZUwyqmS-f"
with open("catboost_trained.pkl", 'wb') as f:
    pickle.dump(boost_model, f)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1642188107612, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="NRchFAmAPdBh" outputId="b3675f02-b241-4c38-8d08-095ffc172f50"
# with open("catboost_trained.pkl", 'rb') as f:
#     boost_model = pickle.load(f)
boost_model
```

```python id="X0yG3zmjPqC1"
random_items = list(np.random.choice(interactions_df['user_id'], size=5, replace=False))
cold_items = [10000, 20000]
random_items.extend(cold_items)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1642188281959, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="JBqnIdlKQFIy" outputId="6e1a8e42-562f-4ad9-a45f-f196fc4e5f74"
warm_idx = np.intersect1d(random_items, interactions_df['user_id'].unique())
warm_idx
```

```python id="xvkws1XHQP1O"
_candidates = candidates.copy()
_candidates.dropna(subset=['item_id'], axis=0, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 488} executionInfo={"elapsed": 1745, "status": "ok", "timestamp": 1642188284831, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="DmhCuWoOQgA4" outputId="38d1ca58-104c-4aca-c15f-3c5f70bc9510"
submit_feat = _candidates.merge(users_df[user_col],
                               on=['user_id'],
                               how='left') \
    .merge(items_df[item_col],
           on=['item_id'],
           how='left')
submit_feat
```

```python id="P00tQC_ZQrYm"
full_train = submit_feat.fillna('None')
full_train[cat_col] = full_train[cat_col].astype('category')
# item_stats = pd.read_csv('data/item_stats_for_submit.csv')
full_train = full_train.join(item_stats.set_index('item_id'),
                             on='item_id', how='left')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 488} executionInfo={"elapsed": 733, "status": "ok", "timestamp": 1642188360258, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="mwa-Fp1wDbW-" outputId="a80d2a9c-dba5-4d17-f339-dea2e3894792"
full_train
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 892, "status": "ok", "timestamp": 1642188385461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="5vqmW3AADiFL" outputId="72dc7b75-7c6a-4d58-f554-01f8a3a76b78"
cols
```

```python colab={"base_uri": "https://localhost:8080/", "height": 488} executionInfo={"elapsed": 1394, "status": "ok", "timestamp": 1642188508132, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="i96FiqviQu42" outputId="679f1b7a-7de9-46b7-a850-ddeeeedcf102"
# Renaming columns to match classifier feature names
cols = ['user_id', 'item_id']
cols.extend(boost_model.feature_names_)
cols = cols[:7] + ['boost_user_watch_cnt_all', 'boost_user_watch_cnt_last_14'] + cols[9:]
full_train = full_train[cols]
full_train_new_names = ['user_id', 'item_id'] + boost_model.feature_names_
full_train.columns = full_train_new_names
full_train
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} executionInfo={"elapsed": 7830, "status": "ok", "timestamp": 1642188520391, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="PdLtTGROQ11b" outputId="5e6dc810-1edb-4565-88af-1b9487ee9372"
# Making predictions for warm users
y_pred_all = boost_model.predict_proba(full_train.drop(
    ['user_id', 'item_id'], axis=1))
full_train['boost_pred'] = y_pred_all[:, 1]
full_train = full_train[['user_id', 'item_id', 'boost_pred']]
full_train = full_train.sort_values(by=['user_id', 'boost_pred'],
                                    ascending=[True, False])
full_train['rank'] = full_train.groupby('user_id').cumcount() + 1
full_train = full_train[full_train['rank'] <= 10].drop('boost_pred', axis=1)
full_train['item_id'] = full_train['item_id'].astype('int64')
boost_recs = full_train.groupby('user_id')['item_id'].apply(list)
boost_recs = pd.DataFrame(boost_recs)
boost_recs.reset_index(inplace=True)
boost_recs
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1642188521208, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="2bCBSBe_57JM" outputId="5328db55-be6d-42fd-c8a3-4857f13c1e32"
# Making predictions for cold users with Popular Recommender
idx_for_popular = list(set(pd.Series(random_items).unique()).difference(
    set(boost_recs['user_id'].unique())))
idx_for_popular
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1642188521956, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="yCctjmQT6AJV" outputId="b2273d78-9891-4580-d238-1598366c3bb7"
interactions_df
```

```python id="tQKdXPRl6C5f"
pop_model = PopularRecommender(days=30, dt_column='last_watch_dt',
                               with_filter=True)
pop_model.fit(interactions_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 238} executionInfo={"elapsed": 8734, "status": "ok", "timestamp": 1642188532800, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="ec2JpRUH6LZv" outputId="66f7fa3a-78dc-4051-b1f4-59a7b374e4dd"
recs_popular = pop_model.recommend_with_filter(interactions_df, idx_for_popular, top_K=10)
recs_popular
```

```python id="KTbSaiIyRBgu"
all_recs = pd.concat([boost_recs, recs_popular], axis=0)
```

```python id="rLyH2YZaShKv"
def fill_with_popular(recs, pop_model_fitted, interactions_df, top_K=10):
    """
    Fills missing recommendations with Popular Recommender.
    Takes top_K first recommendations if length of recs exceeds top_K
    """
    recs['len'] = recs['item_id'].apply(lambda x: len(x))
    recs_good = recs[recs['len'] >= top_K].copy()
    recs_good.loc[(recs_good['len'] > top_K), 'item_id'] = recs_good.loc[
        (recs_good['len'] > 10), 'item_id'].apply(lambda x: x[:10])
    recs_bad = recs[recs['len'] < top_K].copy()
    recs_bad['num_popular'] = top_K - recs_bad.len
    idx_for_filling = recs_bad['user_id'].unique()
    filling_recs = pop_model_fitted.recommend_with_filter(
        interactions_df, idx_for_filling, top_K=top_K)
    recs_bad = recs_bad.join(filling_recs.set_index('user_id'),
                             on='user_id', how='left', rsuffix='1')
    recs_bad.loc[(recs_bad['len'] > 0), 'item_id'] = \
        recs_bad.loc[(recs_bad['len'] > 0), 'item_id'] + \
        recs_bad.loc[(recs_bad['len'] > 0), 'item_id1']
    recs_bad.loc[(recs_bad['len'] == 0), 'item_id'] = recs_bad.loc[
        (recs_bad['len'] == 0), 'item_id1']
    recs_bad['item_id'] = recs_bad['item_id'].apply(lambda x: x[:top_K])
    total_recs = pd.concat([recs_good[['user_id', 'item_id']],
                            recs_bad[['user_id', 'item_id']]], axis=0)
    return total_recs
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} executionInfo={"elapsed": 8980, "status": "ok", "timestamp": 1642188541766, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="397i5e_fmiHS" outputId="eb66e6ee-3a57-451c-9c22-07a629ee8f4b"
# Filling short recommendations woth popular items
all_recs = fill_with_popular(all_recs, pop_model, interactions_df)
all_recs
```

<!-- #region id="unhZ55xCzSII" -->
## Baseline

Popularity based model

Ref: [Official baseline tutorial](https://github.com/recohut/notebooks/blob/main/extras/mts_baseline.ipynb)
<!-- #endregion -->

```python id="LT8NuO96TICh"
def calculate_novelty(train_interactions, recommendations, top_n): 
    users = recommendations['user_id'].unique()
    n_users = train_interactions['user_id'].nunique()
    n_users_per_item = train_interactions.groupby('item_id')['user_id'].nunique()

    recommendations = recommendations.loc[recommendations['rank'] <= top_n].copy()
    recommendations['n_users_per_item'] = recommendations['item_id'].map(n_users_per_item)
    recommendations['n_users_per_item'] = recommendations['n_users_per_item'].fillna(1)
    recommendations['item_novelty'] = -np.log2(recommendations['n_users_per_item'] / n_users)

    item_novelties = recommendations[['user_id', 'rank', 'item_novelty']]
    
    miuf_at_k = item_novelties.loc[item_novelties['rank'] <= top_n, ['user_id', 'item_novelty']]
    miuf_at_k = miuf_at_k.groupby('user_id').agg('mean').squeeze()

    return miuf_at_k.reindex(users).mean()
```

```python id="MujfY8TjTICi"
def compute_metrics(train, test, recs, top_N):
    result = {}
    test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', 'rank'])

    test_recs['users_item_count'] = test_recs.groupby(level='user_id')['rank'].transform(np.size)
    test_recs['reciprocal_rank'] = (1 / test_recs['rank']).fillna(0)
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']
    
    users_count = test_recs.index.get_level_values('user_id').nunique()

    for k in range(1, top_N + 1):
        hit_k = f'hit@{k}'
        test_recs[hit_k] = test_recs['rank'] <= k
        result[f'Precision@{k}'] = (test_recs[hit_k] / k).sum() / users_count
        result[f'Recall@{k}'] = (test_recs[hit_k] / test_recs['users_item_count']).sum() / users_count
        
    result[f'MAP@{top_N}'] = (test_recs['cumulative_rank'] / test_recs['users_item_count']).sum() / users_count
    result[f'Novelty@{top_N}'] = calculate_novelty(train, recs, top_N)
    
    return pd.Series(result)
```

<!-- #region id="P28xd48xTICz" -->
### Example on one fold
<!-- #endregion -->

```python id="uJVqVnskTIC0"
test = interactions_df[interactions_df['last_watch_dt'] == interactions_df['last_watch_dt'].max()]
train = interactions_df[interactions_df['last_watch_dt'] < interactions_df['last_watch_dt'].max()]
```

```python id="yWAZei5ETIC1"
pop_model = PopularRecommender(days=7, dt_column='last_watch_dt')
pop_model.fit(train)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1642188574386, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="t9hL7kFfTIC2" outputId="fbf7c514-37eb-4f69-8510-b71704236b99"
top10_recs = pop_model.recommend()
top10_recs
```

```python id="TKmJ8SydTIC3"
item_titles = pd.Series(items_df['title'].values, index=items_df['item_id']).to_dict()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1642188574389, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="3n6RAbh6TIC4" outputId="6b113f1b-2cbc-4d0f-9c81-bf20fcb16787"
list(map(item_titles.get, top10_recs))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1642188574390, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="lVk4_qpvTIC4" outputId="87000191-3012-4a90-e132-7969e06fa538"
recs = pd.DataFrame({'user_id': test['user_id'].unique()})
top_N = 10
recs['item_id'] = pop_model.recommend(recs['user_id'], N=top_N)
recs.head()
```

```python id="UPjTuarETIC5"
recs = recs.explode('item_id')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 426} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1642188576837, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="XqjGRg15TIC6" outputId="25c15140-b328-4cb9-d875-8f6733f89229"
recs['rank'] = recs.groupby('user_id').cumcount() + 1
recs.head(top_N + 2)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2593, "status": "ok", "timestamp": 1642188579423, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="Yp74HHqnTIC6" outputId="5a94f5ac-7cdd-4bc4-a50e-3ce1d97bdc7a"
compute_metrics(train, test, recs, 10)
```

<!-- #region id="_3I9v8q7UlYk" -->
### Folder validation

Let's take the last 3 weeks from our data and test them sequentially (1 test fold - 1 week). Don't forget about the cold start problem.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 688, "status": "ok", "timestamp": 1642188582610, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="8VkbYiXhTIC8" outputId="19421d57-7726-4866-e826-ec8a7298d36b"
last_date = interactions_df['last_watch_dt'].max().normalize()
folds = 3
start_date = last_date - pd.Timedelta(days=folds*7)
start_date, last_date
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1642188583154, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="1ByOzE3HTIC9" outputId="573e4e37-8d82-4e81-fe1c-986cf0e7a466"
cv = TimeRangeSplit(start_date=start_date, periods=folds+1, freq='W')

cv.max_n_splits, cv.get_n_splits(interactions_df, datetime_column='last_watch_dt')
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1642188585451, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="cwvzIFYcTIC9" outputId="9edcb93e-acb9-406b-9e10-4e019c598275"
cv.date_range
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2877, "status": "ok", "timestamp": 1642188588772, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="6-Ridv5hTIC-" outputId="b3ad2cd1-5a25-4a09-8c61-0bab9774fe19"
folds_with_stats = list(cv.split(
    interactions_df, 
    user_column='user_id',
    item_column='item_id',
    datetime_column='last_watch_dt',
    fold_stats=True
))

folds_info_with_stats = pd.DataFrame([info for _, _, info in folds_with_stats])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 257} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1642188588773, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="pPHoISGQTIC_" outputId="e3595ffb-eaee-4cc4-ad09-e400a4f71543"
folds_info_with_stats
```

<!-- #region id="oMuGmqVBTIC_" -->
### Popular on folds
<!-- #endregion -->

```python id="Q1bXpzaMTIC_"
top_N = 10
last_n_days = 7
```

```python id="Y9guCzRqTIDA"
final_results = []
validation_results = pd.DataFrame()

for train_idx, test_idx, info in folds_with_stats:
    train = interactions_df.loc[train_idx]
    test = interactions_df.loc[test_idx]
        
    pop_model = PopularRecommender(days=last_n_days, dt_column='last_watch_dt')
    pop_model.fit(train)

    recs = pd.DataFrame({'user_id': test['user_id'].unique()})
    recs['item_id'] = pop_model.recommend(recs['user_id'], N=top_N)
    recs = recs.explode('item_id')
    recs['rank'] = recs.groupby('user_id').cumcount() + 1

    fold_result = compute_metrics(train, test, recs, top_N)

    validation_results = validation_results.append(fold_result, ignore_index=True)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1642188603077, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="cyFK7eHhTIDA" outputId="fc4c35a5-3c20-400f-c48d-4690dee2f79b"
validation_results.agg({'MAP@10':'mean', 'Novelty@10':'mean'})
```

<!-- #region id="hcFFZpFATIDA" -->
### Popular Prediction

Let's see if it makes sense to predict the popular depending on the social group
<!-- #endregion -->

```python id="sfqxhgZuTIDB"
train_idx, test_idx, info = folds_with_stats[0]
train = interactions_df.loc[train_idx]
test = interactions_df.loc[test_idx]
date_window_for_popular = train['last_watch_dt'].max() - pd.DateOffset(days=last_n_days)
train_slice = pd.merge(train[train['last_watch_dt'] >= date_window_for_popular], users_df, on='user_id', how='left')
```

<!-- #region id="ydpUgqh6TIDB" -->
we have users without features, so we need to define padding for them
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 270} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1642188603078, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="tH4n5EAnTIDC" outputId="e8d1f2bd-23ef-4b83-b0bb-b91a08f4eac5"
train_slice.head()
```

```python id="b8VbywpUTIDC"
train_slice.fillna({'age':'age_unknown',
                    'sex':'sex_unknown',
                    'income': 'income_unknown',
                    'kids_flg': False
                   }, inplace=True)
```

<!-- #region id="X8edftA_TIDD" -->
For example, you can watch popular by age, gender and presence of children
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 270} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1642188605384, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="D-q6NC2ZTIDD" outputId="57c555d5-544d-4e78-f25b-e13d5b58436f"
train_slice.head()
```

```python id="zFTxZZNBTIDD"
soc_dem_recommendations = train_slice.groupby(
    ['age', 'sex', 'income', 'item_id']
).size().to_frame().reset_index()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1642188606950, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="66aG48_cTIDD" outputId="d8da5e9a-c3c5-4b4c-fbaf-c971ec676690"
soc_dem_recommendations
```

<!-- #region id="MMRvQGlxTIDE" -->
Now you just need to select for each user the most popular top_n objects in his group
<!-- #endregion -->

<!-- #region id="qxY3Q_uETIDE" -->
We can check this option on folds

<!-- #endregion -->

```python id="-Nm6HPYhTIDF"
validation_results = pd.DataFrame()

for train_idx, test_idx, info in folds_with_stats:
    train = interactions_df.loc[train_idx]
    test = interactions_df.loc[test_idx]
    date_window = train['last_watch_dt'].max() - pd.DateOffset(days=last_n_days)
    train_slice = pd.merge(train[train['last_watch_dt'] >= date_window], users_df, on='user_id', how='left')
    
    train_slice.fillna({
        'age':'age_unknown',
        'sex':'sex_unknown',
        'income': 'income_unknown',
        'kids_flg': False
    },inplace=True)
    
    soc_dem_recommendations = train_slice.groupby(
        ['age', 'sex', 'income', 'item_id']
    ).size().to_frame().reset_index()
    
    top_soc_dem = []

    for age in soc_dem_recommendations.age.unique():
        for income in soc_dem_recommendations.income.unique():
            for sex in soc_dem_recommendations.sex.unique():
                top_items = soc_dem_recommendations[
                (soc_dem_recommendations.age == age)
                & (soc_dem_recommendations.income == income)
                & (soc_dem_recommendations.sex == sex)].sort_values(0, ascending=False).head(10).item_id.values
                top_soc_dem.append([age, income, sex, top_items])

    top_soc_dem = pd.DataFrame(top_soc_dem, columns = ['age', 'income', 'sex', 'item_id'])
    
    recs = pd.DataFrame({'user_id': test['user_id'].unique()})
    recs = pd.merge(recs[['user_id']], users_df, on='user_id', how='left')
    recs.fillna({
        'age':'age_unknown',
        'sex':'sex_unknown',
        'income': 'income_unknown',
        'kids_flg': False
    }, inplace=True)
    
    recs = pd.merge(recs, top_soc_dem, on = ['age', 'sex', 'income'], how = 'left')
    recs = recs.drop(columns = ['age', 'sex', 'income'])
    
    recs = recs.explode('item_id')
    recs['rank'] = recs.groupby('user_id').cumcount() + 1
    fold_result = compute_metrics(train, test, recs, top_N)
    
    validation_results = validation_results.append(fold_result, ignore_index=True)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1642188624221, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="k81s04vuTIDF" outputId="b6d5d848-4ce9-4b88-aa4f-294094d0396c"
validation_results.agg({'MAP@10':'mean', 'Novelty@10':'mean'})
```

<!-- #region id="8vbHhlUiTIDG" -->
In this case, the features by which you build the popular are selected, as well as the number of days that you take to calculate the popular
<!-- #endregion -->

<!-- #region id="OPv2gBKETIDG" -->
### Tfidf
<!-- #endregion -->

```python id="pitmMjB2TIDH"
users_inv_mapping = dict(enumerate(interactions_df['user_id'].unique()))
users_mapping = {v: k for k, v in users_inv_mapping.items()}

items_inv_mapping = dict(enumerate(interactions_df['item_id'].unique()))
items_mapping = {v: k for k, v in items_inv_mapping.items()}
```

```python id="MQtorexITIDH"
validation_results = pd.DataFrame()

for train_idx, test_idx, info in folds_with_stats:
    train = interactions_df.loc[train_idx]

    date_window = train['last_watch_dt'].max() - pd.DateOffset(days=60)
    train = train[train['last_watch_dt'] >= date_window]

    test = interactions_df.loc[test_idx]

    train_mat = get_coo_matrix(
        train,
        users_mapping=users_mapping,
        items_mapping=items_mapping,
    ).tocsr()

    model = TFIDFRecommender(K=top_N)
    model.fit(train_mat.T, show_progress=False) 

    mapper = generate_implicit_recs_mapper( 
        model,
        train_mat,
        top_N,
        users_mapping,
        items_inv_mapping,
        filter_already_liked_items=True
    )

    recs = pd.DataFrame({'user_id': test['user_id'].unique()})
    recs['item_id'] = recs['user_id'].map(mapper)
    recs = recs.explode('item_id')
    recs['rank'] = recs.groupby('user_id').cumcount() + 1
    fold_result = compute_metrics(train, test, recs, top_N)

    validation_results = validation_results.append(fold_result, ignore_index=True)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1642188699563, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="tm0_BSCLTIDI" outputId="2aa86238-a2d5-4f3a-9190-ea77c87dc56b"
validation_results.agg({'MAP@10':'mean', 'Novelty@10':'mean',})
```

<!-- #region id="dzI0rVytTIDI" -->
Simply using the code above for submission won't work due to cold users. We'll have to figure out how to process them.
<!-- #endregion -->

<!-- #region id="4d54eqKGTIDI" -->
### Predictions
<!-- #endregion -->

```python id="OOswWzXxWxGK"
random_items = list(np.random.choice(interactions_df['user_id'], size=5, replace=False))
cold_items = [10000, 20000]
random_items.extend(cold_items)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1642188699565, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="SAbO_8EFXAgY" outputId="11260a68-a229-48e2-a724-5e5d52019917"
random_items
```

```python id="bZd04bj0TIDJ"
train = interactions_df
test = random_items

pop_model = PopularRecommender(days=last_n_days, dt_column='last_watch_dt')
pop_model.fit(train)

recs = pd.DataFrame({'user_id': pd.Series(test).unique()})
recs['item_id'] = pop_model.recommend(recs['user_id'], N=top_N)
recs = recs.explode('item_id')
recs['rank'] = recs.groupby('user_id').cumcount() + 1
recs = recs.groupby('user_id').agg({'item_id': list}).reset_index()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1642188699568, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="hmb6rWr2TIDJ" outputId="3a1a7d85-1d2a-4be6-993a-c02b902e18f1"
recs.head()
```

<!-- #region id="YT7-dpYKEqub" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 714, "status": "ok", "timestamp": 1642188912940, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="SwzKjSnTFmfa" outputId="490c855f-1a59-440e-8529-0690e102aa2c"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d -p implicit,catboost,recohut
```

<!-- #region id="VaKjWG8IEquj" -->
---
<!-- #endregion -->

<!-- #region id="c1vxSboeEquj" -->
**END**
<!-- #endregion -->
