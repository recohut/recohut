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

```python id="rxL9QQCZPx8Q" executionInfo={"status": "ok", "timestamp": 1630816987288, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-chef"; branch = "30music"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="NGWuLt_QRJ3f" executionInfo={"status": "ok", "timestamp": 1630817014544, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c6c3527a-4d76-4a65-c91e-e7dd2df3eba9"
!git checkout "{branch}"
```

```python id="khIQ-3vH88Yt" executionInfo={"status": "ok", "timestamp": 1630816648761, "user_tz": -330, "elapsed": 1745, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-sess"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="dwLTtRQK88Yz" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630816992276, "user_tz": -330, "elapsed": 4997, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="946809a0-2785-407c-a2f7-3fd15c94e2b4"
if not os.path.exists(project_path):
    !pip install -U -q dvc dvc[gdrive]
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="2jrtc9Bg88Y0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630817448912, "user_tz": -330, "elapsed": 440, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eb1cacf2-73d9-4eb3-a9c3-e548815fd0cf"
!git status
```

```python id="2G4iErkK88Y1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630817459417, "user_tz": -330, "elapsed": 1142, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6fce36a1-365a-46ca-8b91-ff8a22ac25d8"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="wrKpCfvK_0bB"
!dvc pull
```

```python id="aqbmXvVB-_s8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630816987286, "user_tz": -330, "elapsed": 33371, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9ac6b558-68ef-414a-c376-3233e53f9bd7"
!dvc commit && dvc push
```

<!-- #region id="zv6JRDrHAjVz" -->
---
<!-- #endregion -->

```python id="EhvjiARkIQTP"
import numpy as np
import pandas as pd
import datetime
import calendar
import time
from collections import Counter
```

```python id="QuPR1snZKTfj"
class SessionDataset:
    def __init__(self, df, seed=42):
        self.data = df.copy()
        self._standardize()
        self.seed = seed
        self.train = None
        self.test = None

    def _standardize(self):
        col_names = ['session_id', 'user_id', 'item_id', 'ts'] + self.data.columns.values.tolist()[4:]
        self.data.columns = col_names

    def _add_months(self, sourcedate, months):
        month = sourcedate.month - 1 + months
        year = int(sourcedate.year + month / 12)
        month = month % 12 + 1
        day = min(sourcedate.day, calendar.monthrange(year, month)[1])
        return datetime.date(year, month, day)

    def filter_by_time(self, last_months=0):
        if last_months > 0:
            lastdate = datetime.datetime.fromtimestamp(self.data.ts.max())
            firstdate = self._add_months(lastdate, -last_months)
            initial_unix = time.mktime(firstdate.timetuple())
            self.data = self.data[self.data['ts'] >= initial_unix]

    def convert_to_sequence(self, topk=0):
        c = Counter(list(self.data['item_id']))
        if topk > 1:
            keeper = set([x[0] for x in c.most_common(topk)])
            self.data = self.data[self.data['item_id'].isin(keeper)]

        # group by session id and concat song_id
        groups = self.data.groupby('session_id')

        # convert item ids to string, then aggregate them to lists
        aggregated = groups['item_id'].agg(sequence = lambda x: list(map(str, x)))
        init_ts = groups['ts'].min()
        users = groups['user_id'].min()  # it's just fast, min doesn't actually make sense

        self.data = aggregated.join(init_ts).join(users)
        self.data.reset_index(inplace=True)

    def get_stats(self):
        cnt = Counter()
        _stats = []
        self.data.sequence.map(cnt.update);
        sequence_length = self.data.sequence.map(len).values
        n_sessions_per_user = self.data.groupby('user_id').size()

        _stats.append('Number of items: {}'.format(len(cnt)))
        _stats.append('Number of users: {}'.format(self.data.user_id.nunique()))
        _stats.append('Number of sessions: {}'.format(len(self.data)) )

        _stats.append('Session length:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
            sequence_length.mean(), 
            np.quantile(sequence_length, 0.5), 
            sequence_length.min(), 
            sequence_length.max()))

        _stats.append('Sessions per user:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
            n_sessions_per_user.mean(), 
            np.quantile(n_sessions_per_user, 0.5), 
            n_sessions_per_user.min(), 
            n_sessions_per_user.max()))

        _stats.append('Most popular items: {}'.format(cnt.most_common(5)))
        _stats =  '\n'.join(_stats)
        
        return _stats

    def random_holdout(self, split=0.8):
        """
        Split sequence data randomly
        :param split: the training percentange
        """
        self.data = self.data.sample(frac=1, random_state=self.seed)
        nseqs = len(self.data)
        train_size = int(nseqs * split)
        self.train = self.data[:train_size]
        self.test = self.data[train_size:]

    def temporal_holdout(self, ts_threshold):
        """
        Split sequence data using timestamps
        :param ts_threshold: the timestamp from which test sequences will start
        """
        self.train = self.data.loc[self.data['ts'] < ts_threshold]
        self.test = self.data.loc[self.data['ts'] >= ts_threshold]
        self.train, self.test = self._clean_split(self.train, self.test)

    def last_session_out_split(self,
                               user_key='user_id', 
                               session_key='session_id',
                               time_key='ts'):
        """
        Assign the last session of every user to the test set and the remaining ones to the training set
        """
        sessions = self.data.sort_values(by=[user_key, time_key]).groupby(user_key)[session_key]
        last_session = sessions.last()
        self.train = self.data[~self.data.session_id.isin(last_session.values)].copy()
        self.test = self.data[self.data.session_id.isin(last_session.values)].copy()
        self.train, self.test = self._clean_split(self.train, self.test)

    def _clean_split(self, train, test):
        """
        Remove new items from the test set.
        :param train: The training set.
        :param test: The test set.
        :return: The cleaned training and test sets.
        """
        train = train.copy()
        test = test.copy()
        train_items = set()
        train['sequence'].apply(lambda seq: train_items.update(set(seq)))
        test['sequence'] = test['sequence'].apply(lambda seq: [it for it in seq if it in train_items])
        return train, test
```

```python id="Vt-eccBVIOQe"
df = pd.read_parquet('./data/bronze/30music/sessions_sample_10.parquet.snappy')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="uNklfQom19rd" executionInfo={"status": "ok", "timestamp": 1630524664008, "user_tz": -330, "elapsed": 516, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d818be6a-ed28-4963-c8bd-3610c15aa5c4"
df.head()
```

```python id="ZnRsb5CqIgip"
df.info()
```

<!-- #region id="9s9wjS0qI5WU" -->
### Scratch testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 328} id="CxnuEFIM2-Ht" executionInfo={"status": "ok", "timestamp": 1630525183222, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="301b0689-979f-4d39-fe33-7374e1220b81"
test_df = [[1,1,1,'2015-01-13',10],
             [2,1,1,'2015-02-13',20],
             [2,1,3,'2015-02-13',5],
             [3,1,3,'2015-02-14',15],
             [4,2,1,'2014-12-13',10],
             [5,2,2,'2015-02-10',2],
             [5,2,1,'2015-02-10',9],
             [5,2,3,'2015-02-10',3],
             [5,2,3,'2015-02-10',7],
             ]
test_df = pd.DataFrame(test_df)
test_df.columns = ['session_id', 'user_id', 'item_id', 'ts', 'playtime']
test_df.ts = test_df.ts.apply(dt_int)
test_df
```

```python id="rTJWhov3uytR"
xx = SessionDataset(test_df)
# xx.filter_by_time(last_months=0)
# xx.filter_by_time(last_months=1)
xx.convert_to_sequence()
# xx.convert_to_sequence(topk=2)
# xx.get_stats()
# xx.data#.to_dict()

# xx.random_holdout(0.6)
xx.temporal_holdout(1423600000)
# xx.temporal_holdout(1423500000)
# xx.last_session_out_split()
# display(xx.train)
# display(xx.test)
xx.train.to_dict('list')
# xx.train
```

<!-- #region id="CrHvhGF5I8HV" -->
### Unit testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="12c7S2zQPfjh" executionInfo={"status": "ok", "timestamp": 1630529589585, "user_tz": -330, "elapsed": 753, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a67d62e8-c341-4006-ea76-53d442d675aa"
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal


test_df = [[1,1,1,'2015-01-13',10],
             [2,1,1,'2015-02-13',20],
             [2,1,3,'2015-02-13',5],
             [3,1,3,'2015-02-14',15],
             [4,2,1,'2014-12-13',10],
             [5,2,2,'2015-02-10',2],
             [5,2,1,'2015-02-10',9],
             [5,2,3,'2015-02-10',3],
             [5,2,3,'2015-02-10',7],
             ]
test_df = pd.DataFrame(test_df)
test_df.columns = ['session_id', 'user_id', 'item_id', 'ts', 'playtime']


def _dt_int(dt, tm='00:00:00'):
    """converts date (& time) to integer"""
    return int(datetime.datetime.strptime('{} {}'.format(dt,tm), '%Y-%m-%d %H:%M:%S').strftime("%s"))

test_df.ts = test_df.ts.apply(dt_int)


class TestDataset(unittest.TestCase):
    def setUp(self):
        pass

    def testFilterByTimeNoFilter(self):
        """If month=0, do not remove any rows
        passing first n rows of the test_df,
        expected not to remove any rows
        """
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.filter_by_time(last_months=0)
        assert_frame_equal(test_df.iloc[:,:], _dataset.data)

    def testFilterByTimeFilter(self):
        """If month>0, remove rows
        passing first n rows of the test_df,
        expected to remove some rows
        """
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.filter_by_time(last_months=1)
        assert_frame_equal(test_df.iloc[[1,2,3,5,6,7,8],:], _dataset.data)

    def testItemConversionToSequence(self):
        """convert items to a list in time-based sequence
        passing first n rows of the test_df,
        expected as per dictionary frame defined below
        """
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _expecteddf = pd.DataFrame.from_dict({
            'session_id': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
            'sequence': {0: ['1'], 1: ['1', '3'], 2: ['3'], 3: ['1'], 4: ['2', '1', '3', '3']},
            'ts': {0: 1421107200, 1: 1423785600, 2: 1423872000, 3: 1418428800, 4: 1423526400},
            'user_id': {0: 1, 1: 1, 2: 1, 3: 2, 4: 2}})
        assert_frame_equal(_expecteddf, _dataset.data)        

    def testItemConversionToSequenceTopK(self):
        """convert items to a list in time-based sequence
        filters topk most interacted items
        passing first n rows of the test_df with topk=2,
        expected as per dictionary frame defined below
        """
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence(topk=2)
        _expecteddf = pd.DataFrame.from_dict({
            'session_id': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
            'sequence': {0: ['1'], 1: ['1', '3'], 2: ['3'], 3: ['1'], 4: ['1', '3', '3']},
            'ts': {0: 1421107200, 1: 1423785600, 2: 1423872000, 3: 1418428800, 4: 1423526400},
            'user_id': {0: 1, 1: 1, 2: 1, 3: 2, 4: 2}})
        assert_frame_equal(_expecteddf, _dataset.data)   

    def testDataStatistics(self):
        """generate statistics of the dataset
        passing first n rows of the test_df,
        expected as per string defined below
        expected:
        Number of items: 3\nNumber of users: 2\nNumber of sessions: 
        5\nSession length:\n\tAverage: 1.80\n\tMedian: 1.0\n\tMin: 
        1\n\tMax: 4\nSessions per user:\n\tAverage: 2.50\n\tMedian: 
        2.5\n\tMin: 2\n\tMax: 3\nMost popular items: 
        [('1', 4), ('3', 4), ('2', 1)]"""
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _generated = _dataset.get_stats()        
        self.assertIn("Number of items: 3", _generated)    
        self.assertIn("Most popular items: [('1', 4), ('3', 4), ('2', 1)]", _generated)    
        self.assertIn("Session length:\n\tAverage: 1.80\n\tMedian: 1.0", _generated)    
        self.assertNotIn("Number of items: 4", _generated)     

    def testRandomSplit(self):
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _dataset.random_holdout(0.6)
        _expecteddf = pd.DataFrame.from_dict(
            {'session_id': {1: 2, 2: 3, 4: 5},
            'sequence': {1: ['1', '3'], 2: ['3'], 4: ['2', '1', '3', '3']},
            'ts': {1: 1423785600, 2: 1423872000, 4: 1423526400},
            'user_id': {1: 1, 2: 1, 4: 2}}
            )
        _expecteddf = _expecteddf.reindex([1,4,2])
        assert_frame_equal(_expecteddf, _dataset.train)

    def testTemporalSplitThreshold1(self):
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _dataset.temporal_holdout(1423600000)
        _expecteddf = pd.DataFrame.from_dict(
            {'session_id': {0: 1, 3: 4, 4: 5},
            'sequence': {0: ['1'], 3: ['1'], 4: ['2', '1', '3', '3']},
            'ts': {0: 1421107200, 3: 1418428800, 4: 1423526400},
            'user_id': {0: 1, 3: 2, 4: 2}}
            )
        assert_frame_equal(_expecteddf, _dataset.train) 

    def testTemporalSplitThreshold2(self):
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _dataset.temporal_holdout(1423500000)
        _expecteddf = pd.DataFrame.from_dict(
            {'session_id': {0: 1, 3: 4},
            'sequence': {0: ['1'], 3: ['1']},
            'ts': {0: 1421107200, 3: 1418428800},
            'user_id': {0: 1, 3: 2}}
        )
        assert_frame_equal(_expecteddf, _dataset.train) 

    def testSessionOutSplit(self):
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _dataset.last_session_out_split()
        _expecteddf = pd.DataFrame.from_dict(
            {'session_id': {0: 1, 1: 2, 3: 4},
            'sequence': {0: ['1'], 1: ['1', '3'], 3: ['1']},
            'ts': {0: 1421107200, 1: 1423785600, 3: 1418428800},
            'user_id': {0: 1, 1: 1, 3: 2}}
        )
        assert_frame_equal(_expecteddf, _dataset.train)

unittest.main(argv=[''], verbosity=2, exit=False)
```

<!-- #region id="GW66CzEQJe9r" -->
### Library testing
<!-- #endregion -->

```python id="rlg3-v0xPcXX"
import sys
sys.path.insert(0,'./src')
```

```python id="tWwnr0KwJlcC"
from src.data import SessionDataset as SD
```

```python id="m3lZaSzkKEnS"
df = pd.read_parquet('./data/bronze/30music/sessions_sample_10.parquet.snappy')
```

```python id="VkNBkOoNJ9Lp"
sd = SD(df)
```

```python colab={"base_uri": "https://localhost:8080/"} id="bVNRKS2KKIwP" executionInfo={"status": "ok", "timestamp": 1630530107822, "user_tz": -330, "elapsed": 1934, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a8635bec-b42a-4d2d-9289-772edde02502"
sd.convert_to_sequence()
stats = sd.get_stats()
print(stats)
```

<!-- #region id="Ek_a9uur02oB" -->
### Script testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bzG69PlA05L9" executionInfo={"status": "ok", "timestamp": 1630557951848, "user_tz": -330, "elapsed": 866, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3c37da14-5be1-4a28-b130-57cd69a9a0ac"
!make setup
```

```python colab={"base_uri": "https://localhost:8080/"} id="Rj8sI7WO068g" executionInfo={"status": "ok", "timestamp": 1630557999687, "user_tz": -330, "elapsed": 1540, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8caef8ce-2946-4940-9a09-b785b7442e33"
!make test
```

```python id="7oaE8cld1IO1"

```
