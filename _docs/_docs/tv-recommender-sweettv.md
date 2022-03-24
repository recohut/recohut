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

<!-- #region id="Nk7rQKVZuH7f" -->
# SweetTV Recommendation System
> building a tv-show and movie recommender system using lightfm and lightgbm models on sweet.tv's streaming dataset

- toc: true
- badges: true
- comments: true
- categories: [LightFM, HyperOpt, Kaggle, Movie, Streaming]
- image:
<!-- #endregion -->

<!-- #region id="9e9cNJhnUeak" -->
<!-- #endregion -->

<!-- #region id="Pr5yN9ocVG_O" -->
[sweet.tv](https://sweet.tv/en) is a streaming service that offers access to TV-channels, Ukrainian films, world-class movies, cartoons, and the best series
<!-- #endregion -->

<!-- #region id="wN6SZZSirDX2" -->
## Planning
<!-- #endregion -->

<!-- #region id="hJbU1_5WrLZT" -->
<!-- #endregion -->

<!-- #region id="HxxyhJHtqxF2" -->
## Setup
<!-- #endregion -->

```python id="9NcKbY8EmaiL"
!pip install lightfm
```

```python id="idqGxCJCq4ax"
import ast
import tqdm
import scipy
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

import lightgbm as lgb
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k, auc_score
from lightfm.cross_validation import random_train_test_split

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

%matplotlib inline
sns.set_style('whitegrid')
```

```python colab={"base_uri": "https://localhost:8080/"} id="GdizjewFRrq_" outputId="a2cd9771-8a22-4dfb-cf6f-677bfef6dd44"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv
```

<!-- #region id="zguHoz5JVO1f" -->
## Data loading
<!-- #endregion -->

<!-- #region id="ZW44lJ6KVga1" -->
<!-- #endregion -->

```python id="ygd-g5xEVUyp"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c sweettv-tv-program-recommender
```

```python id="-feOG4ALVZoy"
!unzip -qqo /content/sweettv-tv-program-recommender.zip
```

<!-- #region id="oMvqZwyiVk_G" -->
## EDA
<!-- #endregion -->

<!-- #region id="oFUpGnbArZ-b" -->
**dataset11-30week.csv** - the training set. `vsetv_id` is the same as `channel_id` in export_arh_*.csv files.
<!-- #endregion -->

```python id="WUcG0MONd-wd" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="62fba7c2-eba0-4556-d38e-c93811912c49"
df = pd.read_csv('dataset11-30.csv')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="RyRuvRR3elsS" outputId="a0243d97-a2c1-41be-d759-75e6551f7b8b"
df.info()
```

```python id="uXaTbqxLeP8X"
df.start_time = pd.to_datetime(df.start_time)
df.stop_time = pd.to_datetime(df.stop_time)
df['date'] = df['start_time'].dt.date
```

```python colab={"base_uri": "https://localhost:8080/", "height": 306} id="1d2LNs8keQ4o" outputId="c63b8f10-1d06-4015-d6e2-e20cfff8e3f3"
df.describe(include='all', datetime_is_numeric=True).T
```

<!-- #region id="rt4B8r9Cr5i8" -->
**export_arh_*.csv files** - supplemental information about train data (TV program schedule). Contains tv_show_id - have to predict 5 of these for the submission user_ids.
<!-- #endregion -->

```python id="hhYpovGmd-wf" colab={"base_uri": "https://localhost:8080/", "height": 479} outputId="5a42aadb-d92b-4880-bf27-29c3ee52a8d1"
df_info = pd.read_csv('export_arh_11-20-final.csv').append(pd.read_csv('export_arh_21-30-final.csv'))
df_info.start_time = pd.to_datetime(df_info.start_time, format='%d.%m.%Y %H:%M:%S')
df_info['stop_time'] = df_info['start_time'] + pd.to_timedelta(df_info['duration'], unit='s')
df_info = df_info[df_info.tv_show_id != 0].copy().reset_index(drop=True)
df_info.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="LjBepX8Xe_Zx" outputId="6104a551-3442-4a5d-86b7-94ae5caf9eca"
df_info.info()
```

```python id="_ERF3vr1d-wg" colab={"base_uri": "https://localhost:8080/"} outputId="8632bd96-6666-417e-f70b-9e503d468c32"
print('No. of users:', df.user_id.nunique())
print('No. of channels:', df.vsetv_id.nunique())
print('No. of TV shows:', df_info.tv_show_id.nunique())
print('Data dates:', df.start_time.min(), df.start_time.max())
print('Data dates info:', df_info.start_time.min(), df_info.start_time.max())
```

```python id="wbK229Xid-wi" colab={"base_uri": "https://localhost:8080/", "height": 251} outputId="4e1eced5-79c5-4d35-a08f-6fe90ef00af6"
ax = df.groupby(['date']).size().plot(figsize = (20, 7));
ax.set_title('Views by time', fontsize = 15);
```

<!-- #region id="a3sdasa3d-wj" -->
There are many signs in the names of TV programs - cinema, season, series, premiere
<!-- #endregion -->

```python id="O_A-fLXHd-wj"
for c in ['эп.', 'м/с', 'х/ф', 'д/ф', 'м/ф', 'т/с', 'премьера']:
    c_index = df_info[df_info['tv_show_title'].str.lower().str.contains(c)].index

for c in ['c.', 'c..']:
    c_index = df_info[df_info['tv_show_title'].str.lower()==c].index
```

```python id="-_NRu2bHd-wl" colab={"base_uri": "https://localhost:8080/", "height": 235} outputId="27eae634-0af1-4e1c-8d6c-57cb21c5ddfb"
channel_info_df = df_info[['channel_id', 'channel_title']].drop_duplicates()
channel_info_df[channel_info_df.duplicated(['channel_id'], keep = False)].sort_values(['channel_id'])
```

```python id="uMv8bc6ld-wm" colab={"base_uri": "https://localhost:8080/", "height": 332} outputId="8625c4ba-2fb8-4eef-cb5f-a73f5cb7c5a8"
show_info_df = df_info[['tv_show_id', 'tv_show_category', 'tv_show_genre_1', 
                        'tv_show_genre_2', 'tv_show_genre_3', 'year_of_production', 
                        'director', 'actors']].drop_duplicates()
show_info_df[show_info_df.duplicated(['tv_show_id'], keep = False)].sort_values(['tv_show_id']).head(3)
```

```python id="tCMmbvNtd-wn" colab={"base_uri": "https://localhost:8080/", "height": 142} outputId="d8e1228c-09be-4001-bb4f-5e9d371a2506"
show_duration_df = df_info[['tv_show_id', 'duration']].drop_duplicates()
show_duration_df[show_duration_df.duplicated(['tv_show_id'], keep = False)].sort_values(['tv_show_id']).head(3)
```

```python id="MLKAklhAd-wn" colab={"base_uri": "https://localhost:8080/", "height": 322} outputId="94b28686-42a0-4451-9f5e-5fad2a33e936"
ax = df_info['channel_title'].value_counts()[:25].plot(figsize = (20, 7), kind = 'bar');
ax.set_title('Top channels by transmission');
```

```python id="lP5RIGfRd-wo" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="1c36f4b3-ae9c-46ea-8623-2e9ba214eca9"
tv_show_id_df = df_info[['tv_show_id', 'tv_show_title']].drop_duplicates()
tv_show_id_df[tv_show_id_df.duplicated(['tv_show_id'], keep = False)].sort_values(['tv_show_id']).head()
```

<!-- #region id="9cbqHlQmh4YU" -->
## Preprocessing
<!-- #endregion -->

<!-- #region id="FDcv7D7HiKb1" -->
**Algorithm for collecting data on TV program views**
<!-- #endregion -->

```python id="9L3cdFQyh5so"
show_lines = []
for i, row in tqdm(df.iterrows(), total = df.shape[0]):
    
    user_id = row['user_id']
    vsetv_id = row['vsetv_id']
    start_time = row['start_time']
    stop_time = row['stop_time']
    
    # For each user session, we will find those programs that went on the channel that he watched
    shows_remember_df = df_info[(df_info.channel_id == vsetv_id) 
                    & (df_info.start_time < stop_time) 
                    & (df_info.stop_time > start_time)][['tv_show_id', 'start_time', 'stop_time']].copy()
    
    # Remember which programs the user has watched
    for j, row_j in shows_remember_df.iterrows():
        show_lines.append([user_id, vsetv_id, start_time, stop_time, row_j['tv_show_id'], row_j['start_time'], row_j['stop_time']])
        
show_lines_df = pd.DataFrame(show_lines,columns = ['user_id', 'vsetv_id', 'start_time', 'stop_time', 'tv_show_id', 'show_start_time', 'show_stop_time'])
show_lines_df.to_csv('dataset_with_shows.csv', index = False)
```

<!-- #region id="TjcLCiA_o8b5" -->
## Feature engineering
<!-- #endregion -->

```python id="vAAXgzGdo-wH"
def prepare_features(train, train_info, train_info_future, train_base):
    
    '''
    Function to enrich data by adding features
    '''

    final_shape = train.shape[0]
    
    # collect prediction rank
    for user, group in train.sort_values(['user_id', 'user_show_freq'], ascending = False).groupby(['user_id']):
        train.loc[group.index, 'user_show_rank'] = np.arange(0, group.shape[0])
    train.loc[train[train.user_show_freq == 0].index, 'user_show_rank'] = 50

    # collect prediction rank
    for user, group in train.sort_values(['user_id', 'lightfm_score'], ascending = False).groupby(['user_id']):
        train.loc[group.index, 'user_lfm_rank'] = np.arange(0, group.shape[0])
    train.loc[train[pd.isnull(train.lightfm_score)].index, 'user_lfm_rank'] = 50

    # mean rank
    train['combined_rank'] = (train['user_show_rank'] + train['user_lfm_rank']) / 2

    # tv show channel
    _temp = train_info[['tv_show_id', 'channel_id']].drop_duplicates()
    _temp = _temp[~_temp.duplicated(['tv_show_id'], keep = 'first')].copy()
    train = train.merge(_temp, on = ['tv_show_id'], how = 'left')
    assert final_shape == train.shape[0]

    # add cat ids
    _temp = train_info[['tv_show_id', 'tv_show_category', 'tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3']].drop_duplicates()
    _temp = _temp[~_temp.duplicated(['tv_show_id'], keep = 'first')].copy()
    train = train.merge(_temp, on = ['tv_show_id'], how = 'left')
    assert final_shape == train.shape[0]
    
    # number of watches with diff thresholds
    for threshold in [0.3, 0.8]:
        c_new = f'user_show_freq_{threshold}'
        features_threshold = get_features(train_base, threshold=threshold)
        features_threshold.rename(columns = {'user_show_freq':c_new}, inplace = True)
        train = train.merge(features_threshold, on = ['user_id', 'tv_show_id'], how = 'left')
        train[c_new].fillna(0, inplace = True)
        assert final_shape == train.shape[0]
        
    # alternative base
    total_user_show_watch_df = train_base.groupby(['tv_show_id', 'show_start_time', 'show_stop_time', 'user_id', 'show_duration'], as_index = False).user_watch_time.sum()
    total_user_show_watch_df['user_watch_perc'] = total_user_show_watch_df['user_watch_time'] / total_user_show_watch_df['show_duration']
    total_user_show_watch_df = total_user_show_watch_df[total_user_show_watch_df.user_watch_perc <= 1].copy()
    
    # number of watches with diff thresholds
    for threshold in [0.3, 0.5, 0.8]:
        c_new = f'alt_user_show_freq_{threshold}'
        features_threshold = get_features(total_user_show_watch_df, threshold=threshold)
        features_threshold.rename(columns = {'user_show_freq':c_new}, inplace = True)
        train = train.merge(features_threshold, on = ['user_id', 'tv_show_id'], how = 'left')
        train[c_new].fillna(0, inplace = True)
        assert final_shape == train.shape[0]
    

    # number of watches with more recent time splits
    for weeks_prior in [1]:
        c_new = f'user_show_freq_week_{weeks_prior}'
        split_date = train_base.start_time.max() - datetime.timedelta(days = weeks_prior * 7)
        features_split = get_features(train_base[train_base.start_time >= split_date], 200)
        features_split.rename(columns = {'user_show_freq':c_new}, inplace = True)
        train = train.merge(features_split, on = ['user_id', 'tv_show_id'], how = 'left')
        train[c_new].fillna(0, inplace = True)
        assert final_shape == train.shape[0]
        
        train[f'user_show_freq_dif_week_{weeks_prior}'] = (train['user_show_freq'] - train[c_new]) / train['user_show_freq']
        assert final_shape == train.shape[0]
        
        # new ranks
        for user, group in train.sort_values(['user_id', f'user_show_freq_week_{weeks_prior}'], ascending = False).groupby(['user_id']):
            train.loc[group.index, f'user_show_rank_week_{weeks_prior}'] = np.arange(0, group.shape[0])
        train[f'user_show_rank_mean_week_{weeks_prior}'] = train[['user_show_rank', f'user_show_rank_week_{weeks_prior}']].mean(axis = 1)
        assert final_shape == train.shape[0]
        
        train.drop(c_new, 1, inplace = True)
    
    # number of watches with more recent time splits
    for weeks_prior in [1]:
        c_new = f'alt_user_show_freq_week_{weeks_prior}'
        split_date = total_user_show_watch_df.show_start_time.max() - datetime.timedelta(days = weeks_prior * 7)
        features_split = get_features(total_user_show_watch_df[total_user_show_watch_df.show_start_time >= split_date], 200)
        features_split.rename(columns = {'user_show_freq':c_new}, inplace = True)
        train = train.merge(features_split, on = ['user_id', 'tv_show_id'], how = 'left')
        train[c_new].fillna(0, inplace = True)

        train[f'alt_user_show_freq_dif_week_{weeks_prior}'] = (train['user_show_freq'] - train[c_new]) / train['user_show_freq']

        # new ranks
        for user, group in train.sort_values(['user_id', c_new], ascending = False).groupby(['user_id']):
            train.loc[group.index, f'user_show_rank_week_{weeks_prior}'] = np.arange(0, group.shape[0])
        train[f'alt_user_show_rank_mean_week_{weeks_prior}'] = train[['user_show_rank', f'user_show_rank_week_{weeks_prior}']].mean(axis = 1)

        train.drop(c_new, 1, inplace = True)

    # Насколько часто пользователь смотрит канал
    _temp = train.groupby(['user_id', 'channel_id']).size().reset_index().rename(columns = {0:'user_channel_count'})
    _temp = _temp.merge(_temp.groupby(['user_id'], as_index = False)['user_channel_count'].sum().rename(columns = {'user_channel_count':'user_count'}),
                on = ['user_id'], how = 'left')
    train = train.merge(_temp[['user_id', 'channel_id', 'user_channel_count']],
                on = ['user_id', 'channel_id'], how = 'left')
    
    # user gruop watch mean time+ rel to every watch
    train = train.merge(train.groupby(['user_id'], as_index = False)['user_show_freq'].mean().rename(columns = {'user_show_freq':'group_user_show_freq'}),
                on = ['user_id'], how = 'left')
    train['user_show_freq_rel_group'] = train['user_show_freq'] / train['group_user_show_freq']
    assert final_shape == train.shape[0]

    # show total duration in the future and relative to previous
    _temp_1 = train_info.groupby(['tv_show_id'], as_index = False)['duration'].sum().rename(columns = {'duration':'tot_show_duration'})
    num_days = (train_info.start_time.max() - train_info.start_time.min()).days
    _temp_1['tot_show_duration'] /= num_days
    _temp_2 = train_info_future.groupby(['tv_show_id'], as_index = False)['duration'].sum().rename(columns = {'duration':'tot_show_duration_future'})
    num_days = (train_info_future.start_time.max() - train_info_future.start_time.min()).days
    _temp_2['tot_show_duration_future'] /= num_days
    train = train.merge(_temp_1, on = ['tv_show_id'], how = 'left')
    train = train.merge(_temp_2, on = ['tv_show_id'], how = 'left')
    train['tot_show_duration'].fillna(0, inplace = True)
    train['tot_show_duration_future'].fillna(0, inplace = True)
    train['popularity_drop'] = train['tot_show_duration_future'] / train['tot_show_duration']
    assert final_shape == train.shape[0]
    
    return train
```

<!-- #region id="5nP59BKLpGmN" -->
## Evaluation metrics
<!-- #endregion -->

```python id="PrBLS3FAoomS"
def apk(actual, predicted, k=5):
    
    '''
    Function to get Average Precision at K
    '''
    
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=5):
    
    '''
    Function to get Mean Average Precision at K
    '''
    
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
```

<!-- #region id="g2TGp97Jj3So" -->
## Utils
<!-- #endregion -->

```python id="jiad0vvKj9x1"
user_col = 'cont_user_id'
movie_col = 'cont_tv_show_id'
```

```python id="2IN9Na-zpT6F"
def get_features(df, get_n = 5, threshold = 0.5):
    
    '''
    Function to get target-like features by threshold
    '''
    
    # filter by 80% watch time
    df = df[df['user_watch_perc'] > threshold].copy()
    
    # calc how many times user viewed the show
    df_views = df[['user_id', 'tv_show_id']].groupby(['user_id', 'tv_show_id']).size().reset_index()\
        .rename(columns = {0:'user_show_freq'})
    
    # get only top-5 for each user
    df_top_views = df_views.sort_values(['user_id', 'user_show_freq'], ascending=False).groupby(['user_id']).head(get_n)
    
    return df_top_views
```

```python id="YF44scFhpVnH"
def get_target(df, get_n = 5):
    
    '''
    Function to get target
    '''
    
    # filter by 80% watch time
    df = df[df['user_watch_perc'] >= 0.8].copy()
    
    # calc how many times user viewed the show
    df_views = df[['user_id', 'tv_show_id']].groupby(['user_id', 'tv_show_id']).size().reset_index()\
        .rename(columns = {0:'user_show_freq'})
    
    # get only top-5 for each user
    df_top_views = df_views.sort_values(['user_id', 'user_show_freq'], ascending=False).groupby(['user_id']).head(get_n)
    
    return df_top_views
```

```python id="cG6fMvEmj4Ha"
def df_to_sparse_interaction_matrix(x, has_seen = True):
    '''
    Pandas dataframe to LightFM format
    '''

    interaction_x = x[[user_col, movie_col]].drop_duplicates().assign(seen=1).\
                    pivot_table(index = user_col, columns = movie_col).fillna(0)
        
    return scipy.sparse.csr_matrix(interaction_x)
```

<!-- #region id="oqCl_X1ukQed" -->
## Baseline model
<!-- #endregion -->

<!-- #region id="8i_0k8JOrjwp" -->
<!-- #endregion -->

<!-- #region id="PQRnphIbsmnD" -->
**export_arh_31-42-final.csv** - supplemental information about test data (TV program schedule)
<!-- #endregion -->

```python id="ZhrCoNRSkR66"
df_info_future = pd.read_csv('export_arh_31-42-final.csv', low_memory=False)
```

```python id="w-_hxZ1OkZF0"
df_show = pd.read_csv('dataset_with_shows.csv')
for c in ['start_time', 'show_start_time', 'stop_time','show_stop_time']:
    df_show[c] = pd.to_datetime(df_show[c])
df_show['start_show_user_time'] = df_show[['show_start_time','start_time']].max(axis=1)
df_show['stop_show_user_time'] = df_show[['stop_time','show_stop_time']].min(axis=1)
df_show['user_watch_time'] = (df_show['stop_show_user_time'] - df_show['start_show_user_time']).dt.total_seconds()
df_show['show_duration'] = (df_show['show_stop_time'] - df_show['show_start_time']).dt.total_seconds()
df_show['user_watch_perc'] = df_show['user_watch_time'] / df_show['show_duration']
```

```python id="RajFnAsGkbZN"
# split on holdout by time
train_start_date = df_show.start_time.min()# + datetime.timedelta(days = 4 * 7)
split_date = df_show.start_time.max() - datetime.timedelta(days = 9 * 7)
train = df_show[(df_show.start_time <= split_date) & (df_show.start_time >= train_start_date)].copy()
val = df_show[df_show.start_time > split_date].copy()
```

```python id="UBGAw9QOkbV0"
# collection of signs and target by views
train_top_views = get_features(train[train['tv_show_id'].isin(val['tv_show_id'].unique())])
val_top_views = get_target(val)
overall_top_movies = train_top_views.groupby(['tv_show_id'], as_index = False)['user_show_freq'].sum().sort_values(['user_show_freq'])['tv_show_id'][-5:].values
overall_top_movies = overall_top_movies[::-1]
```

```python colab={"base_uri": "https://localhost:8080/"} id="WqGh9k85kbR5" outputId="49a96966-4fd5-480d-fc10-3a714cdff055"
# checking the accuracy of the solution
preds = []
trues = []
for user in train_top_views.user_id.unique():
    
    predict_n_movies = train_top_views[train_top_views.user_id == user]['tv_show_id'].values[:5]
    actual_n_movies = val_top_views[val_top_views.user_id == user]['tv_show_id'].values[:5]
    
    if len(predict_n_movies) < 5:
        predict_n_movies = list(predict_n_movies[:len(predict_n_movies)]) + list(overall_top_movies[:5 - len(predict_n_movies)])
    
    
    preds.append(list(predict_n_movies))
    trues.append(list(actual_n_movies))
    
score = np.round(mapk(trues, preds, k = 5), 5)
print(f'MAP@{5} = {score}')
```

```python id="B4wvqPLQkbOQ"
# view data
df_top_views = get_features(df_show[df_show['tv_show_id'].isin(df_info_future['tv_show_id'].unique())])
overall_top_movies = df_top_views.groupby(['tv_show_id'], as_index = False)['user_show_freq'].sum().sort_values(['user_show_freq'])['tv_show_id'][-5:].values[::-1]

# forecast for the future
submission_df = pd.read_csv('submission.csv')
for index_row, row in submission_df.iterrows():
    pred_n_movies = list(df_top_views[df_top_views.user_id == row['user_id']]['tv_show_id'].values[:5])
    
    if len(pred_n_movies) < 5:
        pred_n_movies = list(pred_n_movies[:len(pred_n_movies)]) + list(overall_top_movies[:5 - len(pred_n_movies)])
    
    pred = ' '.join([str(int(x)) for x in pred_n_movies])
    submission_df.loc[index_row, 'tv_show_id'] = pred
    
submission_df.to_csv('baseline_submission.csv', index = False)
```

<!-- #region id="cU8kVB2hkbKo" -->
## Final model
<!-- #endregion -->

<!-- #region id="fthmfzcMrt6B" -->
<!-- #endregion -->

```python id="K_bxpN57j0JH"
# data from the past
df_info = pd.read_csv('export_arh_11-20-final.csv').append(pd.read_csv('export_arh_21-30-final.csv'))
df_info.start_time = pd.to_datetime(df_info.start_time, format='%d.%m.%Y %H:%M:%S')
df_info['stop_time'] = df_info['start_time'] + pd.to_timedelta(df_info['duration'], unit='s')
df_info = df_info[df_info.tv_show_id != 0].copy()

# data for the future
df_info_future = pd.read_csv('export_arh_31-42-final.csv', low_memory=False)
df_info_future.start_time = pd.to_datetime(df_info_future.start_time, format='%d.%m.%Y %H:%M:%S')
df_info_future['stop_time'] = df_info_future['start_time'] + pd.to_timedelta(df_info_future['duration'], unit='s')
df_info_future = df_info_future[df_info_future.tv_show_id != 0].copy()

# categorical features
for c in ['tv_show_category', 'tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3']:
    df_info[c] = LabelEncoder().fit_transform(df_info[c].fillna('Nope'))
```

<!-- #region id="zArLX4bBj0JK" -->
### User_id, tv_show_id encoding for LightFM
<!-- #endregion -->

```python id="jCJloemlj0JK"
for c in ['user_id', 'tv_show_id']:
    _temp = df_show[[c]].drop_duplicates().reset_index(drop = True)
    _temp[f"cont_{c}"] = np.arange(_temp.shape[0])

    df_show = df_show.merge(_temp, on = [c], how = 'left')
    
user_col = 'cont_user_id'
movie_col = 'cont_tv_show_id'

_temp = df_show[['tv_show_id', movie_col]].drop_duplicates()
movies_dict = dict(zip(_temp[movie_col].values, _temp['tv_show_id'].values))

_temp = df_show[['user_id', user_col]].drop_duplicates()
users_dict = dict(zip(_temp[user_col].values, _temp['user_id'].values))
```

<!-- #region id="7yqRiJGUj0JL" -->
### Checking the solution accuracy on holdout sample
<!-- #endregion -->

<!-- #region id="jxpRoTeXj0JM" -->
**Split data to train / val**
<!-- #endregion -->

```python id="kZ2vHhmvj0JM" colab={"base_uri": "https://localhost:8080/"} outputId="0a53e9c1-51ed-4e1a-e28a-17d5861d7abf"
train_start_date = df_show.start_time.min()
split_date = df_show.start_time.max() - datetime.timedelta(days = 9 * 7) # 9 weeks

train = df_show[(df_show.start_time <= split_date) & (df_show.start_time >= train_start_date)].copy()
df_info_train = df_info[(df_info.start_time <= split_date) & (df_info.start_time >= train_start_date)].copy()

val = df_show[df_show.start_time > split_date].copy()
df_info_val = df_info[df_info.start_time > split_date].copy()

# We remember which TV programs can be validated
possible_movies_in_val = df_info_val.tv_show_id.unique()
print(train.shape, val.shape)
```

<!-- #region id="OuN9J4Gaj0JN" -->
**Train LightFm**
<!-- #endregion -->

```python id="qe8XCgQKj0JN" colab={"base_uri": "https://localhost:8080/", "height": 100} outputId="41c09d0d-df27-4f0e-d207-988b2e22771c"
# sparse for lightfm
train_sparse = df_to_sparse_interaction_matrix(train)

# fit lightfm
model = LightFM(random_state=42, loss = 'warp')
model.fit(train_sparse, epochs=15, num_threads=4, verbose = False);

# collect user biases
user_bias_df = pd.DataFrame(model.user_biases, columns = ['user_bias_lfm'])
user_bias_df['user_id'] = list(sorted(train[user_col].unique()))
user_bias_df['user_id'] = user_bias_df['user_id'].apply(lambda x: users_dict.get(x))

# collect item biases
show_bias_df = pd.DataFrame(model.item_biases, columns = ['show_bias_lfm'])
show_bias_df['tv_show_id'] = list(sorted(train[movie_col].unique()))
show_bias_df['tv_show_id'] = show_bias_df['tv_show_id'].apply(lambda x: movies_dict.get(x))

# predict the top N current programs for each user from train
train_users = sorted(train[user_col].unique())

# we can only predict films that will definitely be in the target
train_movies = sorted(train[train['tv_show_id'].isin(possible_movies_in_val)][movie_col].unique())

get_n = 200
lightfm_predictions_df = pd.DataFrame()
for train_user in train_users:
    
    all_movies_df = pd.DataFrame(train_movies, columns = [movie_col])
    all_movies_df[user_col] = train_user
    all_movies_df = all_movies_df.astype('int32')
    all_movies_df['lightfm_score'] = model.predict(all_movies_df[user_col].values,
                                                   all_movies_df[movie_col].values)
    all_movies_df.sort_values('lightfm_score', ascending = False, inplace = True)
    all_movies_df['lightfm_rank'] = np.arange(all_movies_df.shape[0])
    
    lightfm_predictions_df = lightfm_predictions_df.append(all_movies_df.head(get_n))
    
lightfm_predictions_df['user_id'] = lightfm_predictions_df[user_col].map(lambda x: users_dict.get(x))
lightfm_predictions_df['tv_show_id'] = lightfm_predictions_df[movie_col].map(lambda x: movies_dict.get(x))
lightfm_predictions_df.head(1)
```

<!-- #region id="v04z57D3j0JR" -->
**Choice of N movies for user based on views and LightFm results**
<!-- #endregion -->

```python id="c9d2gVqyj0JR" colab={"base_uri": "https://localhost:8080/"} outputId="a6530d8e-7614-46e1-b9d4-1f93e5ec3c81"
get_n = 200
train_base = train[train['tv_show_id'].isin(possible_movies_in_val)].copy()
val_base = val.copy()
train = get_features(train_base, get_n)
val = get_target(val_base, get_n)
overal_val = get_target(val_base, 10000)
print(train.shape, val.shape)
```

<!-- #region id="RSs1DNrKj0JS" -->
**Add LightFm as additional lines**
<!-- #endregion -->

```python id="GkhMyHmEj0JT"
# if incomplete recommendations (user watched <get_n) then add recommendations from lightfm
train_w_lfm = train.copy()
for user, group in train.groupby(['user_id']):
    
    if group.shape[0] < get_n:
        
        need_to_add = get_n - group.shape[0]
        add_tv_show_ids_from_lightfm = lightfm_predictions_df[
            (lightfm_predictions_df.user_id == user)&
            (~lightfm_predictions_df.tv_show_id.isin(group.tv_show_id.unique()))]\
            .tv_show_id.values[:need_to_add]
        
        add_df = pd.DataFrame(add_tv_show_ids_from_lightfm, columns = ['tv_show_id'])
        add_df['user_id'] = user
        add_df['user_show_freq'] = 0
        add_df['user_id'] = add_df['user_id'].astype('uint64')
        
        train_w_lfm = train_w_lfm.append(add_df[['user_id', 'tv_show_id', 'user_show_freq']])
        
assert train.user_id.nunique() == train_w_lfm.user_id.nunique()
assert train_w_lfm.groupby(['user_id']).size().min() == get_n

train_w_lfm = train_w_lfm.merge(lightfm_predictions_df[['user_id', 'tv_show_id', 'lightfm_score']],
            on = ['user_id', 'tv_show_id'], how = 'left')

# combined train
train = train_w_lfm.copy()
train.reset_index(inplace = True, drop = True)
```

<!-- #region id="PBZ0PJzTj0JU" -->
**Add target**
<!-- #endregion -->

```python id="30BxGEWAj0JU" colab={"base_uri": "https://localhost:8080/"} outputId="5f9470d6-b77d-475e-fd1d-2470ae08a95f"
# create target from tv programs that user actually saw in the next 9 weeks
target_col = 'seen'
train = train.merge(overal_val.drop(['user_show_freq'], 1).assign(seen=1), on = ['user_id', 'tv_show_id'], how = 'left')
train[target_col].fillna(0, inplace = True)
print('Target distribution:')
train[target_col].value_counts()
```

<!-- #region id="Mak1pO3Tj0JU" -->
**Add signs**
<!-- #endregion -->

```python id="Rfhzrb2sj0JV"
# Remove TV programs that have been watched by less than 10 users
shows_watches = train_base[train_base['user_watch_perc'] >= 0.8].groupby(['tv_show_id']).size().reset_index().rename(columns = {0:'show_watched'})
min_watch = 10
shows_watches = shows_watches[shows_watches['show_watched'] > min_watch].copy()
possible_movies_that_matter = shows_watches.tv_show_id.unique()
train = train[train.tv_show_id.isin(possible_movies_that_matter)].copy()
train.reset_index(inplace = True, drop = True)
```

```python id="UR-ReL0zj0JV"
# collecting signs
train = prepare_features(train, df_info_train, df_info_val, train_base)
train = train.merge(user_bias_df, on = ['user_id'], how = 'left')
train = train.merge(show_bias_df, on = ['tv_show_id'], how = 'left')
```

<!-- #region id="W1r1HxrXj0JV" -->
**Model validation**
<!-- #endregion -->

```python id="AMKJx4CNj0JW" colab={"base_uri": "https://localhost:8080/"} outputId="e89671b3-ffec-46eb-b081-f5c3b944eb5d"
cat_columns = ['tv_show_category', 'tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3']
users_train, users_test = train_test_split(train.user_id.unique(), random_state = 42)

X_train = train[train.user_id.isin(users_train)].copy()
X_test = train[train.user_id.isin(users_test)].copy()
y_train = X_train.pop(target_col)
y_test = X_test.pop(target_col)

X_train = X_train.set_index(["user_id", "tv_show_id"])
X_test = X_test.set_index(["user_id", "tv_show_id"])
print(X_train.shape, X_test.shape)
print('Features:', list(X_train.columns))
```

```python id="A9boEza2j0JW" colab={"base_uri": "https://localhost:8080/"} outputId="460b4cd8-009a-42a3-cd61-a4cdfc5ee68d"
train_data = lgb.Dataset(X_train, y_train)
test_data = lgb.Dataset(X_test, y_test)

param = {
    'task': 'train',
    'objective': 'binary', 
    'metric': 'auc',
    'bagging_fraction': 0.8, 
    'bagging_freq': 4, 
    'colsample_bytree': '0.6', 
    'feature_fraction': 0.75, 
    'learning_rate': 0.01,
    'metric': 'auc',
    'min_data_in_leaf': 20, 
    'num_leaves': 150, 
    'num_threads': 4,
    'reg_alpha': 0.4,
    'reg_lambda': 0.32, 
    'seed': 42,
}

res = {}
bst = lgb.train(
    param, train_data, 
    valid_sets=[train_data, test_data], 
    valid_names=["train", "valid"],
    categorical_feature = cat_columns,
    num_boost_round=10000, evals_result=res, 
    verbose_eval=100, early_stopping_rounds=15)
```

```python id="ZCFvFhDHj0JZ" colab={"base_uri": "https://localhost:8080/"} outputId="6ccf2bae-77d3-4852-f59c-43bd585a3bc6"
X_test["lgb_score"] = bst.predict(X_test, num_iteration=bst.best_iteration)
lgb_res = X_test.reset_index([0, 1])[["user_id",
                                        "tv_show_id",
                                        "lgb_score"]].sort_values("lgb_score",
                                                                  ascending=False)

preds = []
trues = []
for user, group in lgb_res.groupby("user_id"):
    predict_n_movies = list(group.tv_show_id)[:5]
    actual_n_movies = val[val.user_id == user]['tv_show_id'].values[:5]
    
    preds.append(list(predict_n_movies))
    trues.append(list(actual_n_movies))
    
score = np.round(mapk(trues, preds, k = 5), 5)
print(f'Model MAP@{5} = {score}')

# reference (baseline)
preds = []
trues = []
for user, group in lgb_res.groupby("user_id"):
    predict_n_movies = train[train.user_id == user]['tv_show_id'].values[:5]
    actual_n_movies = val[val.user_id == user]['tv_show_id'].values[:5]
    
    preds.append(list(predict_n_movies))
    trues.append(list(actual_n_movies))
    
score = np.round(mapk(trues, preds, k = 5), 5)
print(f'Baseline MAP@{5} = {score}')
```

```python id="ZBZTslx4j0Jb" colab={"base_uri": "https://localhost:8080/", "height": 380} outputId="9ec5e08d-5ea3-412a-e241-0060c94305ba"
lgb.plot_importance(bst, figsize = (15, 10));
```

<!-- #region id="I_9Z4wdkj0Jc" -->
### Final model training
<!-- #endregion -->

```python id="BFiDxonaj0Jc" colab={"base_uri": "https://localhost:8080/"} outputId="aa913fb3-9488-4275-acdc-d29bf9665143"
# Full Train
X_train = train.copy()
y_train = X_train.pop(target_col)

X_train = X_train.set_index(["user_id", "tv_show_id"])
train_data = lgb.Dataset(X_train, y_train)

final_model = lgb.train(
    param, train_data, 
    categorical_feature = cat_columns,
    num_boost_round=bst.best_iteration+100, verbose_eval=False)

final_model.save_model('tuned_.txt')
```

<!-- #region id="lgpyz5m-sR1T" -->
### Implementation details
<!-- #endregion -->

<!-- #region id="vLU9ZKwvsWuD" -->
<!-- #endregion -->

<!-- #region id="xjP_FRqZj0Jd" -->
### Collection of final data and forecast submission
<!-- #endregion -->

<!-- #region id="6u6q-GPtj0Jd" -->
**Update trained Lightfm model, now with all data**
<!-- #endregion -->

```python id="ksfc2zF2j0Jd" colab={"base_uri": "https://localhost:8080/", "height": 100} outputId="92a43430-b2a4-4022-bc05-196fea4c37d5"
# sparse for lightfm
df_sparse = df_to_sparse_interaction_matrix(df_show)

# fit lightfm
lfm_final_model = LightFM(random_state=42, loss = 'warp')
lfm_final_model.fit(df_sparse, epochs=15, num_threads=4, verbose = False);

# predict the top N current programs for each user from train
train_users = sorted(df_show[user_col].unique())

# we can only predict films that will definitely be in the target
train_movies = sorted(df_show[df_show.tv_show_id.isin(df_info_future['tv_show_id'].unique())][movie_col].unique())

# collect user biases
df_user_bias_df = pd.DataFrame(lfm_final_model.user_biases, columns = ['user_bias_lfm'])
df_user_bias_df['user_id'] = list(sorted(df_show[user_col].unique()))
df_user_bias_df['user_id'] = df_user_bias_df['user_id'].apply(lambda x: users_dict.get(x))

# collect item biases
df_show_bias_df = pd.DataFrame(lfm_final_model.item_biases, columns = ['show_bias_lfm'])
df_show_bias_df['tv_show_id'] = list(sorted(df_show[movie_col].unique()))
df_show_bias_df['tv_show_id'] = df_show_bias_df['tv_show_id'].apply(lambda x: movies_dict.get(x))

# collect Lightfm recommendations for all users in the train
get_n = 200
df_lightfm_predictions_df = pd.DataFrame()
for train_user in train_users:
    
    all_movies_df = pd.DataFrame(train_movies, columns = [movie_col])
    all_movies_df[user_col] = train_user
    all_movies_df = all_movies_df.astype('int32')
    all_movies_df['lightfm_score'] = lfm_final_model.predict(all_movies_df[user_col].values,
                                                             all_movies_df[movie_col].values)
    all_movies_df.sort_values('lightfm_score', ascending = False, inplace = True)
    all_movies_df['lightfm_rank'] = np.arange(all_movies_df.shape[0])
    
    df_lightfm_predictions_df = df_lightfm_predictions_df.append(all_movies_df.head(get_n), sort = False)
    
df_lightfm_predictions_df['user_id'] = df_lightfm_predictions_df[user_col].map(lambda x: users_dict.get(x))
df_lightfm_predictions_df['tv_show_id'] = df_lightfm_predictions_df[movie_col].map(lambda x: movies_dict.get(x))
assert df_lightfm_predictions_df.user_id.nunique() == len(train_users)
df_lightfm_predictions_df.head(1)
```

<!-- #region id="4CPe546Cj0Je" -->
**Collecting basic features, enriching with recommendations from LightFM**
<!-- #endregion -->

```python id="YNmR8GeXj0Je" colab={"base_uri": "https://localhost:8080/"} outputId="e7fb86f1-c420-4b11-dae9-e2d1b303c263"
# Basic recommendations - according to the top views
print(df_show.shape)
get_n = 200

possible_test_movies = df_info_future['tv_show_id'].unique()
df_show_base = df_show[df_show['tv_show_id'].isin(possible_test_movies)].copy()
df_show_train = get_features(df_show_base, get_n)
overall_top_movies = df_show_train.groupby(['tv_show_id'], as_index = False)['user_show_freq'].sum().sort_values(['user_show_freq'])['tv_show_id'][-5:].values[::-1]
```

```python id="0r1pWTPsj0Je"
# if incomplete recommendations (user watched <get_n) then add recommendations from lightfm
df_train_w_lfm = df_show_train.copy()
for user, group in df_show_train.groupby(['user_id']):
    
    if group.shape[0] < get_n:
        
        need_to_add = get_n - group.shape[0]
        add_tv_show_ids_from_lightfm = df_lightfm_predictions_df[
            (df_lightfm_predictions_df.user_id == user)&
            (~df_lightfm_predictions_df.tv_show_id.isin(group.tv_show_id.unique()))]\
            .tv_show_id.values[:need_to_add]
        
        add_df = pd.DataFrame(add_tv_show_ids_from_lightfm, columns = ['tv_show_id'])
        add_df['user_id'] = user
        add_df['user_show_freq'] = 0
        add_df['user_id'] = add_df['user_id'].astype('uint64')
        
        df_train_w_lfm = df_train_w_lfm.append(add_df[['user_id', 'tv_show_id', 'user_show_freq']])
        
assert df_show_train.user_id.nunique() == df_train_w_lfm.user_id.nunique()
assert df_train_w_lfm.groupby(['user_id']).size().min() == get_n

df_train_w_lfm = df_train_w_lfm.merge(df_lightfm_predictions_df[['user_id', 'tv_show_id', 'lightfm_score']],
            on = ['user_id', 'tv_show_id'], how = 'left')

# combined train
df_show_train = df_train_w_lfm.copy()
df_show_train.reset_index(inplace = True, drop = True)
```

<!-- #region id="ltWqBGd4j0Jf" -->
**Post-processing and adding features**
<!-- #endregion -->

```python id="DbXYTcHhj0Jf" colab={"base_uri": "https://localhost:8080/"} outputId="5b71fa4e-a74d-41b1-deb7-baa0b02238bc"
# remove low-count movies
shows_watches = df_show_base[df_show_base['user_watch_perc'] >= 0.8].groupby(['tv_show_id']).size().reset_index().rename(columns = {0:'show_watched'})
min_watch = 10
shows_watches = shows_watches[shows_watches['show_watched'] > 10].copy()
possible_movies_that_matter = shows_watches.tv_show_id.unique()
df_show_train = df_show_train[df_show_train.tv_show_id.isin(possible_movies_that_matter)].copy()
df_show_train.reset_index(inplace = True, drop = True)

# collect features
df_show_train = prepare_features(df_show_train, df_info, df_info_future, df_show_base)
df_show_train = df_show_train.merge(df_user_bias_df, on = ['user_id'], how = 'left')
df_show_train = df_show_train.merge(df_show_bias_df, on = ['tv_show_id'], how = 'left')
print(df_show_train.shape)
```

<!-- #region id="qOKMgeG5j0Jg" -->
**Prediction submission**
<!-- #endregion -->

```python id="0E8mSyDnj0Jh"
# predict rank score
X_predict = df_show_train.copy()
X_predict = X_predict.set_index(["user_id", "tv_show_id"])[X_train.columns]
X_predict["lgb_score"] = final_model.predict(X_predict, num_iteration=bst.best_iteration)
```

```python id="sT2WdCQGj0Jh"
pred_res = X_predict.reset_index([0, 1])[["user_id",
                                        "tv_show_id",
                                        "lgb_score"]].sort_values("lgb_score",
                                                                  ascending=False)
```

```python id="GrJO3CyLj0Ji" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="5c7aa6d8-1dba-47fb-9c50-db01631bb49f"
# create submission
submission_df = pd.read_csv('submission.csv')
for index_row, row in submission_df.iterrows():
    pred_n_movies = list(pred_res[pred_res.user_id == row['user_id']]['tv_show_id'].values[:5])
    
    if len(pred_n_movies) < 5:
        pred_n_movies = list(pred_n_movies[:len(pred_n_movies)]) + list(overall_top_movies[:5 - len(pred_n_movies)])
    
    pred = ' '.join([str(int(x)) for x in pred_n_movies])
    submission_df.loc[index_row, 'tv_show_id'] = pred
    
submission_df.to_csv('lfm_lgb6_submission.csv', index = False)
submission_df.head()
```

<!-- #region id="xm1x6wdqpmRE" -->
## References
1. https://github.com/ndmel/2nd_place_recsys_cinema_challenge_2020 `code`
2. https://www.kaggle.com/c/sweettv-tv-program-recommender/overview `data`
3. [Google Image search service](https://www.google.com/search?q=sweet.tv+site&rlz=1C1GCEA_enIN909IN909&sxsrf=ALeKk00uJsjmdu5iUacItcKSdysJYoih4w:1626998276475&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj0tMby8PfxAhUf63MBHY7TCnEQ_AUoA3oECAEQBQ&biw=1366&bih=657) `site`
4. [Sweet.tv official site](https://sweet.tv/) `site`
5. [Google Translate service](https://translate.google.co.in/) `api`
6. [Microsoft PPT Translation service](https://translator.microsoft.com/) `api`
<!-- #endregion -->
