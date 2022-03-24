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

<!-- #region id="vDggSKqsu0hl" -->
# Preprocessing of LastFM Session Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="iJEYIPVK84J1" executionInfo={"status": "ok", "timestamp": 1637925995665, "user_tz": -330, "elapsed": 118868, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b0bed86a-6982-4127-f4b7-70da135a0af8"
!wget -q --show-progress http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
```

```python id="l2kmqYBL9tFu"
!gunzip lastfm-dataset-1K.tar.gz
```

```python colab={"base_uri": "https://localhost:8080/"} id="qho8wxJBAEol" executionInfo={"status": "ok", "timestamp": 1637926102583, "user_tz": -330, "elapsed": 29189, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4c2b0f99-6070-4df0-c3f2-12c28039b0aa"
!tar -xvf lastfm-dataset-1K.tar
```

```python id="Z_Z0Er9bAL2E"
!mv lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv .
!rm -r lastfm-dataset-1K
```

```python id="FCHZ4IWt9KMl"
import pandas as pd
from pandas import Timedelta
import numpy as np
import os
```

```python id="AGHuxB0R9YSX"
def get_session_id(df, interval):
    df_prev = df.shift()
    is_new_session = (df.userId != df_prev.userId) | (
        df.timestamp - df_prev.timestamp > interval
    )
    session_id = is_new_session.cumsum() - 1
    return session_id


def group_sessions(df, interval):
    sessionId = get_session_id(df, interval)
    df = df.assign(sessionId=sessionId)
    return df


def filter_short_sessions(df, min_len=2):
    session_len = df.groupby('sessionId', sort=False).size()
    long_sessions = session_len[session_len >= min_len].index
    df_long = df[df.sessionId.isin(long_sessions)]
    return df_long


def filter_infreq_items(df, min_support=5):
    item_support = df.groupby('itemId', sort=False).size()
    freq_items = item_support[item_support >= min_support].index
    df_freq = df[df.itemId.isin(freq_items)]
    return df_freq


def filter_until_all_long_and_freq(df, min_len=2, min_support=5):
    while True:
        df_long = filter_short_sessions(df, min_len)
        df_freq = filter_infreq_items(df_long, min_support)
        if len(df_freq) == len(df):
            break
        df = df_freq
    return df


def truncate_long_sessions(df, max_len=20, is_sorted=False):
    if not is_sorted:
        df = df.sort_values(['sessionId', 'timestamp'])
    itemIdx = df.groupby('sessionId').cumcount()
    df_t = df[itemIdx < max_len]
    return df_t


def update_id(df, field):
    labels = pd.factorize(df[field])[0]
    kwargs = {field: labels}
    df = df.assign(**kwargs)
    return df


def remove_immediate_repeats(df):
    df_prev = df.shift()
    is_not_repeat = (df.sessionId != df_prev.sessionId) | (df.itemId != df_prev.itemId)
    df_no_repeat = df[is_not_repeat]
    return df_no_repeat


def reorder_sessions_by_endtime(df):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    df_endtime = endtime.sort_values().reset_index()
    oid2nid = dict(zip(df_endtime.sessionId, df_endtime.index))
    sessionId_new = df.sessionId.map(oid2nid)
    df = df.assign(sessionId=sessionId_new)
    df = df.sort_values(['sessionId', 'timestamp'])
    return df


def keep_top_n_items(df, n):
    item_support = df.groupby('itemId', sort=False).size()
    top_items = item_support.nlargest(n).index
    df_top = df[df.itemId.isin(top_items)]
    return df_top


def split_by_time(df, timedelta):
    max_time = df.timestamp.max()
    end_time = df.groupby('sessionId').timestamp.max()
    split_time = max_time - timedelta
    train_sids = end_time[end_time < split_time].index
    df_train = df[df.sessionId.isin(train_sids)]
    df_test = df[~df.sessionId.isin(train_sids)]
    return df_train, df_test


def train_test_split(df, test_split=0.2):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    endtime = endtime.sort_values()
    num_tests = int(len(endtime) * test_split)
    test_session_ids = endtime.index[-num_tests:]
    df_train = df[~df.sessionId.isin(test_session_ids)]
    df_test = df[df.sessionId.isin(test_session_ids)]
    return df_train, df_test


def save_sessions(df, filepath):
    df = reorder_sessions_by_endtime(df)
    sessions = df.groupby('sessionId').itemId.apply(lambda x: ','.join(map(str, x)))
    sessions.to_csv(filepath, sep='\t', header=False, index=False)


def save_dataset(df_train, df_test):
    # filter items in test but not in train
    df_test = df_test[df_test.itemId.isin(df_train.itemId.unique())]
    df_test = filter_short_sessions(df_test)

    print(f'No. of Clicks: {len(df_train) + len(df_test)}')
    print(f'No. of Items: {df_train.itemId.nunique()}')

    # update itemId
    train_itemId_new, uniques = pd.factorize(df_train.itemId)
    df_train = df_train.assign(itemId=train_itemId_new)
    oid2nid = {oid: i for i, oid in enumerate(uniques)}
    test_itemId_new = df_test.itemId.map(oid2nid)
    df_test = df_test.assign(itemId=test_itemId_new)

    print(f'saving dataset to {os.getcwd()}')
    save_sessions(df_train, 'train.txt')
    save_sessions(df_test, 'test.txt')
    num_items = len(uniques)
    with open('num_items.txt', 'w') as f:
        f.write(str(num_items))
```

```python id="lNT5o8TP9D1w"
def preprocess_lastfm(csv_file, usecols, interval, n):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        sep='\t',
        header=None,
        names=['userId', 'timestamp', 'itemId'],
        usecols=usecols,
        parse_dates=['timestamp'],
        infer_datetime_format=True,
    )
    print('start preprocessing')
    df = df.dropna()
    df = update_id(df, 'userId')
    df = update_id(df, 'itemId')
    df = df.sort_values(['userId', 'timestamp'])

    df = group_sessions(df, interval)
    df = remove_immediate_repeats(df)
    df = truncate_long_sessions(df, is_sorted=True)
    df = keep_top_n_items(df, n)
    df = filter_until_all_long_and_freq(df)
    df_train, df_test = train_test_split(df, test_split=0.2)
    save_dataset(df_train, df_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="b1T4444t860G" executionInfo={"status": "ok", "timestamp": 1637926248398, "user_tz": -330, "elapsed": 123494, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a8ee8a34-3486-4760-fb57-e392f4ae29d9"
csv_file = 'userid-timestamp-artid-artname-traid-traname.tsv'
usecols = [0, 1, 2]
interval = Timedelta(hours=8)
n = 40000
preprocess_lastfm(csv_file, usecols, interval, n)
```

<!-- #region id="vTiZQip1-AyA" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="BCQ3dPfa-AyB" executionInfo={"status": "ok", "timestamp": 1637926253484, "user_tz": -330, "elapsed": 5116, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c6eaa108-1dbe-4402-ab9e-cef4c0fc0739"
!apt-get -qq install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="4-opNjd--AyB" executionInfo={"status": "ok", "timestamp": 1637926254122, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6c6256f0-a3ac-46f4-e147-bb54c5802185"
!tree -h --du .
```

```python colab={"base_uri": "https://localhost:8080/"} id="AMFugpAf-AyB" executionInfo={"status": "ok", "timestamp": 1637926258498, "user_tz": -330, "elapsed": 4386, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="17e0df59-a207-434f-babb-98610ac7dddc"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="RMX2OkNY-AyC" -->
---
<!-- #endregion -->

<!-- #region id="J8VmzGsU-AyC" -->
**END**
<!-- #endregion -->
