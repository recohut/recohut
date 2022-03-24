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

```python colab={"base_uri": "https://localhost:8080/"} id="p9K5Gnt00PjY" executionInfo={"status": "ok", "timestamp": 1637923636296, "user_tz": -330, "elapsed": 3045, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="99e115ed-cf93-4ff2-e5bf-748a92fe541a"
!wget -q --show-progress https://github.com/RecoHut-Datasets/diginetica/raw/main/train-item-views.csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="C1czoR8h1-P1" executionInfo={"status": "ok", "timestamp": 1637923637468, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1a0dae49-6539-444a-c620-81aaa0d25c26"
!head train-item-views.csv
```

<!-- #region id="sTS_iB_m1A48" -->
## Method 1
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="iADt8Lu03Mwy" executionInfo={"status": "ok", "timestamp": 1637923751244, "user_tz": -330, "elapsed": 1434, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="62a73613-827c-445e-ebe0-3ca5be9f719b"
df = pd.read_csv('/content/train-item-views.csv', sep=';')
df.head()
```

```python id="9mPmTptN08qx"
import time
import csv
import pickle
import operator
import datetime
import os
```

```python id="RHQ5V-Ke08oH"
class DigineticaDataset:
    def __init__(self, path='.'):
        self.path = path

    def preprocess(self):
        dataset = os.path.join(self.path, 'train-item-views.csv')
        print("-- Starting @ %ss" % datetime.datetime.now())
        with open(dataset, "r") as f:
            reader = csv.DictReader(f, delimiter=';')
            sess_clicks = {}
            sess_date = {}
            ctr = 0
            curid = -1
            curdate = None
            for data in reader:
                sessid = data['sessionId']
                if curdate and not curid == sessid:
                    date = ''
                    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                    sess_date[curid] = date
                curid = sessid
                item = data['itemId'], int(data['timeframe'])
                curdate = ''
                curdate = data['eventdate']
                if sessid in sess_clicks:
                    sess_clicks[sessid] += [item]
                else:
                    sess_clicks[sessid] = [item]
                ctr += 1
            date = ''
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            for i in list(sess_clicks):
                sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
                sess_clicks[i] = [c[0] for c in sorted_clicks]
            sess_date[curid] = date

        print("-- Reading data @ %ss" % datetime.datetime.now())

        # Filter out length 1 sessions
        for s in list(sess_clicks):
            if len(sess_clicks[s]) == 1:
                del sess_clicks[s]
                del sess_date[s]

        # Count number of times each item appears
        iid_counts = {}
        for s in sess_clicks:
            seq = sess_clicks[s]
            for iid in seq:
                if iid in iid_counts:
                    iid_counts[iid] += 1
                else:
                    iid_counts[iid] = 1

        sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

        length = len(sess_clicks)
        for s in list(sess_clicks):
            curseq = sess_clicks[s]
            filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
            if len(filseq) < 2:
                del sess_clicks[s]
                del sess_date[s]
            else:
                sess_clicks[s] = filseq

        # Split out test set based on dates
        dates = list(sess_date.items())
        maxdate = dates[0][1]

        for _, date in dates:
            if maxdate < date:
                maxdate = date

        # 7 days for test
        splitdate = 0
        splitdate = maxdate - 86400 * 7

        print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
        tra_sess = filter(lambda x: x[1] < splitdate, dates)
        tes_sess = filter(lambda x: x[1] > splitdate, dates)

        # Sort sessions by date
        tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
        tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
        print(len(tra_sess))    # 186670    # 7966257
        print(len(tes_sess))    # 15979     # 15324
        print(tra_sess[:3])
        print(tes_sess[:3])
        
        print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

        # Choosing item count >=5 gives approximately the same number of items as reported in paper
        item_dict = {}
        # Convert training sessions to sequences and renumber items to start from 1
        def obtian_tra():
            train_ids = []
            train_seqs = []
            train_dates = []
            item_ctr = 1
            for s, date in tra_sess:
                seq = sess_clicks[s]
                outseq = []
                for i in seq:
                    if i in item_dict:
                        outseq += [item_dict[i]]
                    else:
                        outseq += [item_ctr]
                        item_dict[i] = item_ctr
                        item_ctr += 1
                if len(outseq) < 2:  # Doesn't occur
                    continue
                train_ids += [s]
                train_dates += [date]
                train_seqs += [outseq]
            print(item_ctr)     # 43098, 37484
            return train_ids, train_dates, train_seqs


        # Convert test sessions to sequences, ignoring items that do not appear in training set
        def obtian_tes():
            test_ids = []
            test_seqs = []
            test_dates = []
            for s, date in tes_sess:
                seq = sess_clicks[s]
                outseq = []
                for i in seq:
                    if i in item_dict:
                        outseq += [item_dict[i]]
                if len(outseq) < 2:
                    continue
                test_ids += [s]
                test_dates += [date]
                test_seqs += [outseq]
            return test_ids, test_dates, test_seqs

        tra_ids, tra_dates, tra_seqs = obtian_tra()
        tes_ids, tes_dates, tes_seqs = obtian_tes()

        def process_seqs(iseqs, idates):
            out_seqs = []
            out_dates = []
            labs = []
            ids = []
            for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
                for i in range(1, len(seq)):
                    tar = seq[-i]
                    labs += [tar]
                    out_seqs += [seq[:-i]]
                    out_dates += [date]
                    ids += [id]
            return out_seqs, out_dates, labs, ids

        tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
        te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
        tra = (tr_seqs, tr_labs)
        tes = (te_seqs, te_labs)
        print(len(tr_seqs))
        print(len(te_seqs))
        print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
        print(te_seqs[:3], te_dates[:3], te_labs[:3])
        all = 0

        for seq in tra_seqs:
            all += len(seq)
        for seq in tes_seqs:
            all += len(seq)
        print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))


        pickle.dump(tra, open('train.txt', 'wb'))
        pickle.dump(tes, open('test.txt', 'wb'))
        pickle.dump(tra_seqs, open('all_train_seq.txt', 'wb'))

        print('Done.')
```

```python colab={"base_uri": "https://localhost:8080/"} id="WM2P5vMu1n6n" executionInfo={"status": "ok", "timestamp": 1637923823771, "user_tz": -330, "elapsed": 19703, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0ece8e8b-9489-48d5-c272-e36cce8f11a0"
yc_data = DigineticaDataset(path='.')
yc_data.preprocess()
```

<!-- #region id="vlXCHf3T14sM" -->
## Method 2
<!-- #endregion -->

```python id="YAe1mHdg3mOj"
import pandas as pd
import numpy as np
```

```python id="2joR-cij3oSH"
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

```python id="kndB2edY4ATb"
def preprocess_diginetica(csv_file):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        usecols=[0, 2, 3, 4],
        delimiter=';',
        parse_dates=['eventdate'],
        infer_datetime_format=True,
    )
    print('start preprocessing')
    # timeframe (time since the first query in a session, in milliseconds)
    df['timestamp'] = pd.to_timedelta(df.timeframe, unit='ms') + df.eventdate
    df = df.drop(['eventdate', 'timeframe'], 1)
    df = df.sort_values(['sessionId', 'timestamp'])
    df = filter_short_sessions(df)
    df = truncate_long_sessions(df, is_sorted=True)
    df = filter_infreq_items(df)
    df = filter_short_sessions(df)
    df_train, df_test = split_by_time(df, pd.Timedelta(days=7))
    save_dataset(df_train, df_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Su2jIQKw4GnK" executionInfo={"status": "ok", "timestamp": 1637924039014, "user_tz": -330, "elapsed": 10467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2208ccab-1afb-4f71-e334-e12ed108a3a4"
preprocess_diginetica('train-item-views.csv')
```

<!-- #region id="kRSd66q-v8bT" -->
## Method 3
<!-- #endregion -->

<!-- #region id="PkNByPwowEri" -->
https://github.com/kunwuz/DGTN/blob/master/preprocess/cikm16_org_prepro.py
<!-- #endregion -->

```python id="05rycybwtrgI"
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import os
import pickle
import time
```

```python id="TfaZ1Uw7twYw"
raw_path = 'train-item-views'
save_path = 'unaugment_data'
```

```python id="NSVO6oQHt_FD"
def load_data(file):
    print("Start load_data")
    # load csv
    # data = pd.read_csv(file+'.csv', sep=';', header=0, usecols=[0, 2, 4], dtype={0: np.int32, 1: np.int64, 3: str})
    data = pd.read_csv(file + '.csv', sep=';', header=0, usecols=[0, 2, 3, 4],
                       dtype={0: np.int32, 1: np.int64, 2: str, 3: str})
    # specify header names
    # data.columns = ['SessionId', 'ItemId', 'Eventdate']
    data.columns = ['SessionId', 'ItemId', 'Timeframe', 'Eventdate']
    # convert time string to timestamp and remove the original column
    data['Time'] = data.Eventdate.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
    print(data['Time'].min())
    print(data['Time'].max())
    del(data['Eventdate'])

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    return data
```

```python id="kXUhkQ7-uI0s"
def filter_data(data, min_item_support=5, min_session_length=2):
    print("Start filter_data")

    # y?
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]

    # filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= min_item_support].index)]

    # filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= min_session_length].index)]
    print(data['Time'].min())
    print(data['Time'].max())
    # output
    data_start = datetime.fromtimestamp(data.Time.astype(np.int64).min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.astype(np.int64).max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    return data
```

```python id="yBqe_G2ZuLfY"
def split_train_test(data):
    print("Start split_train_test")
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-7*86400].index
    session_test = session_max_times[session_max_times >= tmax-7*86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))

    return train, test
```

```python id="N02x1X30uNzZ"
def get_dict(data):
    print("Start get_dict")
    item2idx = {}
    pop_scores = data.groupby('ItemId').size().sort_values(ascending=False)
    pop_scores = pop_scores / pop_scores[:1].values[0]
    items = pop_scores.index
    for idx, item in enumerate(items):
        item2idx[item] = idx+1
    return item2idx
```

```python id="sTwklY9OuRPm"
def process_seqs(seqs, shift):
    start = time.time()
    labs = []
    index = shift
    for count, seq in enumerate(seqs):
        index += (len(seq) - 1)
        labs += [index]
        end = time.time()
        print("\rprocess_seqs: [%d/%d], %.2f, usetime: %fs, " % (count, len(seqs), count/len(seqs) * 100, end - start),
              end='', flush=True)
    print("\n")
    return seqs, labs
```

```python id="tl6Zr-x-uSnk"
def get_sequence(data, item2idx, shift=-1):
    start = time.time()
    sess_ids = data.drop_duplicates('SessionId', 'first')
    print(sess_ids)
    sess_ids.sort_values(['Time'], inplace=True)
    sess_ids = sess_ids['SessionId'].unique()
    seqs = []
    for count, sess_id in enumerate(sess_ids):
        # seq = data[data['SessionId'].isin([sess_id])]
        seq = data[data['SessionId'].isin([sess_id])].sort_values(['Timeframe'])
        seq = seq['ItemId'].values
        outseq = []
        for i in seq:
            if i in item2idx:
                outseq += [item2idx[i]]
        seqs += [outseq]
        end = time.time()
        print("\rGet_sequence: [%d/%d], %.2f , usetime: %fs" % (count, len(sess_ids), count/len(sess_ids) * 100, end - start),
              end='', flush=True)

    print("\n")
    # print(seqs)
    out_seqs, labs = process_seqs(seqs, shift)
    # print(out_seqs)
    # print(labs)
    print(len(out_seqs), len(labs))
    return out_seqs, labs
```

```python id="FrdiDVTcuWyD"
def preprocess(train, test, path=save_path):
    print("--------------")
    print("Start preprocess cikm16")
    # print("Start preprocess sample")
    item2idx = get_dict(train)
    train_seqs, train_labs = get_sequence(train, item2idx)
    test_seqs, test_labs = get_sequence(test, item2idx, train_labs[-1])
    train = (train_seqs, train_labs)
    test = (test_seqs, test_labs)
    path = path + '/cikm16'
    # path = path + '/sample'
    if not os.path.exists(path):
        os.makedirs(path)

    pickle.dump(test, open(path+'/unaug_test.txt', 'wb'))
    pickle.dump(train, open(path+'/unaug_train.txt', 'wb'))
    print("finished")
```

```python colab={"base_uri": "https://localhost:8080/"} id="tl9tQIl0uhF_" outputId="1080cc91-77c3-4c84-e2ae-96f5a4759e45"
data = load_data(raw_path)
data = filter_data(data)
train, test = split_train_test(data)
preprocess(train, test)
```

<!-- #region id="ajD8rxjf4s8g" -->
---
<!-- #endregion -->

```python id="QnwX1Dsl4s8j"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="mqBQEAS14s8k" -->
---
<!-- #endregion -->

<!-- #region id="NxoXCcpd4s8k" -->
**END**
<!-- #endregion -->
