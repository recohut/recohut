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

<!-- #region id="PZao1y_hJVdT" -->
# Preprocessing of Music Sequential Dataset
<!-- #endregion -->

<!-- #region id="p9n_Vxwo9ZAe" -->
### Imports
<!-- #endregion -->

```python id="eRUde2qi9PGC"
import os
import time
import datetime
import calendar
from collections import Counter
import numpy as np
import pandas as pd
```

<!-- #region id="c2p-pRIK9YvG" -->
## Utils
<!-- #endregion -->

```python id="n6ese-ao9Yr5"
def get_test_sequences(test_data, given_k):
    # we can run evaluation only over sequences longer than abs(LAST_K)
    test_sequences = test_data.loc[test_data['sequence'].map(len) > abs(given_k), 'sequence'].values
    return test_sequences
```

```python id="pF7w3EEO9Ymz"
def get_test_sequences_and_users(test_data, given_k, train_users):
    # we can run evaluation only over sequences longer than abs(LAST_K)
    mask = test_data['sequence'].map(len) > abs(given_k)
    mask &= test_data['user_id'].isin(train_users)
    test_sequences = test_data.loc[mask, 'sequence'].values
    test_users = test_data.loc[mask, 'user_id'].values
    return test_sequences, test_users
```

<!-- #region id="9DcKXyIk9Yhz" -->
## Data
<!-- #endregion -->

```python id="onoh6T1b9Yd9"
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
zipurl = 'https://github.com/RecoHut-Datasets/30music/raw/v2/sessions.zip'
with urlopen(zipurl) as zipresp:
    with ZipFile(BytesIO(zipresp.read())) as zfile:
        zfile.extractall('datasets')
```

```python id="mxfUguSV-ozx"
dataset_path = 'datasets/sessions.csv'
# load this sample if you experience a severe slowdown with the previous dataset
dataset_path = 'datasets/sessions_sample_10.csv'
```

```python id="vZT3lzbP-gl8"
def load_and_adapt(path, last_months=0):
    file_ext = os.path.splitext(path)[-1]
    if file_ext == '.csv':
        data = pd.read_csv(path, header=0)
    elif file_ext == '.hdf':
        data = pd.read_hdf(path)
    else:
        raise ValueError('Unsupported file {} having extension {}'.format(path, file_ext))

    col_names = ['session_id', 'user_id', 'item_id', 'ts'] + data.columns.values.tolist()[4:]
    data.columns = col_names

    if last_months > 0:
        def add_months(sourcedate, months):
            month = sourcedate.month - 1 + months
            year = int(sourcedate.year + month / 12)
            month = month % 12 + 1
            day = min(sourcedate.day, calendar.monthrange(year, month)[1])
            return datetime.date(year, month, day)

        lastdate = datetime.datetime.fromtimestamp(data.ts.max())
        firstdate = add_months(lastdate, -last_months)
        initial_unix = time.mktime(firstdate.timetuple())

        # filter out older interactions
        data = data[data['ts'] >= initial_unix]

    return data
```

```python id="lDWV6brM-twk"
def create_seq_db_filter_top_k(path, topk=0, last_months=0):
    file = load_and_adapt(path, last_months=last_months)

    c = Counter(list(file['item_id']))

    if topk > 1:
        keeper = set([x[0] for x in c.most_common(topk)])
        file = file[file['item_id'].isin(keeper)]

    # group by session id and concat song_id
    groups = file.groupby('session_id')

    # convert item ids to string, then aggregate them to lists
    aggregated = groups['item_id'].agg(sequence = lambda x: list(map(str, x)))
    init_ts = groups['ts'].min()
    users = groups['user_id'].min()  # it's just fast, min doesn't actually make sense

    result = aggregated.join(init_ts).join(users)
    result.reset_index(inplace=True)
    return result
```

```python id="1ZEb52jn_AVG"
# for the sake of speed, let's keep only the top-1k most popular items in the last month
dataset = create_seq_db_filter_top_k(path=dataset_path, topk=1000, last_months=1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="8COI0MYG_C4B" executionInfo={"status": "ok", "timestamp": 1638680895276, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="401be767-07ba-4fd2-f241-35e947f58769"
dataset.head()
```

<!-- #region id="CQaqpX6q_oEd" -->
### Statistics
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WMxIz9YK_pRQ" executionInfo={"status": "ok", "timestamp": 1638680965653, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="948077e6-92b2-4150-fbda-8c1ad59e26f7"
cnt = Counter()
dataset.sequence.map(cnt.update);

sequence_length = dataset.sequence.map(len).values
n_sessions_per_user = dataset.groupby('user_id').size()

print('Number of items: {}'.format(len(cnt)))
print('Number of users: {}'.format(dataset.user_id.nunique()))
print('Number of sessions: {}'.format(len(dataset)) )

print('\nSession length:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
    sequence_length.mean(), 
    np.quantile(sequence_length, 0.5), 
    sequence_length.min(), 
    sequence_length.max()))

print('Sessions per user:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
    n_sessions_per_user.mean(), 
    np.quantile(n_sessions_per_user, 0.5), 
    n_sessions_per_user.min(), 
    n_sessions_per_user.max()))

print('Most popular items: {}'.format(cnt.most_common(5)))
```

<!-- #region id="4FrKcyHQ_0jQ" -->
### Splitting
<!-- #endregion -->

```python id="Do4hLcWrADfs"
def random_holdout(dataset, perc=0.8, seed=1234):
    """
    Split sequence dataset randomly
    :param dataset: the sequence dataset
    :param perc: the training percentange
    :param seed: the random seed
    :return: the training and test splits
    """
    dataset = dataset.sample(frac=1, random_state=seed)
    nseqs = len(dataset)
    train_size = int(nseqs * perc)
    # split data according to the shuffled index and the holdout size
    train_split = dataset[:train_size]
    test_split = dataset[train_size:]

    return train_split, test_split
```

```python id="IgQ_d-msAJvj"
def temporal_holdout(dataset, ts_threshold):
    """
    Split sequence dataset using timestamps
    :param dataset: the sequence dataset
    :param ts_threshold: the timestamp from which test sequences will start
    :return: the training and test splits
    """
    train = dataset.loc[dataset['ts'] < ts_threshold]
    test = dataset.loc[dataset['ts'] >= ts_threshold]
    train, test = clean_split(train, test)

    return train, test
```

```python id="WnxsKfAGAI80"
def last_session_out_split(data,
                           user_key='user_id',
                           session_key='session_id',
                           time_key='ts'):
    """
    Assign the last session of every user to the test set and the remaining ones to the training set
    """
    sessions = data.sort_values(by=[user_key, time_key]).groupby(user_key)[session_key]
    last_session = sessions.last()
    train = data[~data.session_id.isin(last_session.values)].copy()
    test = data[data.session_id.isin(last_session.values)].copy()
    train, test = clean_split(train, test)
    return train, test
```

```python id="TiPSG5u0AIIc"
def clean_split(train, test):
    """
    Remove new items from the test set.
    :param train: The training set.
    :param test: The test set.
    :return: The cleaned training and test sets.
    """
    train_items = set()
    train['sequence'].apply(lambda seq: train_items.update(set(seq)))
    test['sequence'] = test['sequence'].apply(lambda seq: [it for it in seq if it in train_items])
    return train, test
```

```python id="0blcsGbNAG8c"
def balance_dataset(x, y):
    number_of_elements = y.shape[0]
    nnz = set(find(y)[0])
    zero = set(range(number_of_elements)).difference(nnz)

    max_samples = min(len(zero), len(nnz))

    nnz_indices = random.sample(nnz, max_samples)
    zero_indeces = random.sample(zero, max_samples)
    indeces = nnz_indices + zero_indeces

    return x[indeces, :], y[indeces, :]
```

<!-- #region id="qdZKc69ZAEhX" -->
For simplicity, let's split the dataset by assigning the last session of every user to the test set, and all the previous ones to the training set.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="s8V2JG7SAFCh" executionInfo={"status": "ok", "timestamp": 1638681073810, "user_tz": -330, "elapsed": 653, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="585bf9f4-152b-45b7-fde4-3739784d193f"
train_data, test_data = last_session_out_split(dataset)
print("Train sessions: {} - Test sessions: {}".format(len(train_data), len(test_data)))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="qgYvyfsZADcF" executionInfo={"status": "ok", "timestamp": 1638681088161, "user_tz": -330, "elapsed": 738, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="66259bd1-ab98-4b0c-a74d-1734bd7795df"
train_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="gj4-yWUgASIU" executionInfo={"status": "ok", "timestamp": 1638681100734, "user_tz": -330, "elapsed": 616, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="512844bf-e0de-47b8-a410-e2cff87a19c0"
train_data.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="puRzRLi8ADYr" executionInfo={"status": "ok", "timestamp": 1638681092906, "user_tz": -330, "elapsed": 479, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c3c9945a-ce87-49fa-e37d-454ef0de8a6a"
test_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="zPspw4E3ADV5" executionInfo={"status": "ok", "timestamp": 1638681109605, "user_tz": -330, "elapsed": 709, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ea480182-5a5c-4048-b9ec-6e738788abb6"
test_data.info()
```

<!-- #region id="73cKWLHLCPfU" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xiJyIAByUdLS" executionInfo={"status": "ok", "timestamp": 1638681224864, "user_tz": -330, "elapsed": 3738, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="31a0b97b-a8df-4af8-ae41-dc6b16806e24"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="LzMW3rygCRea" -->
---
<!-- #endregion -->

<!-- #region id="w3geplVoEum_" -->
**END**
<!-- #endregion -->
