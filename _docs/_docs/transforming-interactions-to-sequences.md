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

<!-- #region id="vuTj-cGojdG4" -->
```
src data
========
196	242	3	881250949
186	302	3	891717742
22	377	1	878887116
244	51	2	880606923
166	346	1	886397596
298	474	4	884182806
115	265	2	881171488
253	465	5	891628467
305	451	3	886324817
6	86	3	883603013
62	257	2	879372434
286	1014	5	879781125
200	222	5	876042340
210	40	3	891035994
224	29	3	888104457

train_data
==========                          seq            target  user
0     [1, 290, 492, 381, 752]    [467, 523, 11]     1
1   [290, 492, 381, 752, 467]    [523, 11, 673]     1
2   [492, 381, 752, 467, 523]   [11, 673, 1046]     1
3    [381, 752, 467, 523, 11]  [673, 1046, 650]     1
4    [752, 467, 523, 11, 673]  [1046, 650, 378]     1
5   [467, 523, 11, 673, 1046]   [650, 378, 180]     1
6   [523, 11, 673, 1046, 650]   [378, 180, 390]     1
7   [11, 673, 1046, 650, 378]   [180, 390, 666]     1
8  [673, 1046, 650, 378, 180]   [390, 666, 513]     1
9  [1046, 650, 378, 180, 390]   [666, 513, 432]     1

test_data
=========                            seq       target  user
0    [633, 657, 1007, 948, 364]  [522, 0, 0]     1
1       [26, 247, 49, 531, 146]   [32, 0, 0]     2
2      [459, 477, 369, 770, 15]  [306, 0, 0]     3
3  [1093, 946, 1101, 690, 1211]  [526, 0, 0]     4
4     [732, 266, 669, 188, 253]  [986, 0, 0]     5
5      [410, 446, 104, 782, 96]   [26, 0, 0]     6
6     [146, 817, 536, 694, 186]  [525, 0, 0]     7
7      [395, 669, 281, 289, 98]  [731, 0, 0]     8
8     [588, 671, 369, 292, 304]  [250, 0, 0]     9
9        [472, 222, 82, 716, 8]  [131, 0, 0]    10
```
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wen76T61hEdx" executionInfo={"status": "ok", "timestamp": 1637163331033, "user_tz": -330, "elapsed": 1094, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a28648e3-c377-4222-a02f-681fcbc37cc1"
!wget -q --show-progress https://github.com/sparsh-ai/general-recsys/raw/T894414/data/v1/u.data
```

```python colab={"base_uri": "https://localhost:8080/"} id="E5XpoBYWisRM" executionInfo={"status": "ok", "timestamp": 1637163391906, "user_tz": -330, "elapsed": 418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="05a1c7e4-4c14-4b4c-9d16-dab958835625"
!head u.data
```

```python id="6Ps2KkyKiurt" executionInfo={"status": "ok", "timestamp": 1637163398211, "user_tz": -330, "elapsed": 420, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import pandas as pd
import pickle
```

```python id="hH0d2wMGiecB" executionInfo={"status": "ok", "timestamp": 1637163533350, "user_tz": -330, "elapsed": 2, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def make_datasets(file, target_counts, seq_counts, isSave=True):

    file_path = file
    names = ['user', 'item', 'rateing', 'timestamps']
    data = pd.read_csv(file_path, header=None, sep='\t', names=names)

    # ReMap item ids
    item_unique = data['item'].unique().tolist()
    item_map = dict(zip(item_unique, range(1,len(item_unique) + 1)))
    item_map[-1] = 0
    all_item_count = len(item_map)
    data['item'] = data['item'].apply(lambda x: item_map[x])

    # ReMap usr ids
    user_unique = data['user'].unique().tolist()
    user_map = dict(zip(user_unique, range(1, len(user_unique) + 1)))
    user_map[-1] = 0
    all_user_count = len(user_map)
    data['user'] = data['user'].apply(lambda x: user_map[x])

    # Get user session
    data = data.sort_values(by=['user','timestamps']).reset_index(drop=True)

    user_sessions = data.groupby('user')['item'].apply(lambda x: x.tolist()) \
        .reset_index().rename(columns={'item': 'item_list'})

    train_users = []
    train_seqs = []
    train_targets = []

    test_users = []
    test_seqs = []
    test_targets = []

    user_all_items = {}

    for index, row in user_sessions.iterrows():
        user = row['user']
        items = row['item_list']
        user_all_items[user] = items

        for i in range(seq_counts,len(items) - target_counts):
            targets = items[i:i+target_counts]
            seqs = items[max(0,i - seq_counts):i]

            train_users.append(user)
            train_seqs.append(seqs)
            train_targets.append(targets)

        last_item = [items[-1],0,0]
        last_seq = items[-1 - seq_counts:-1]
        test_users.append(user)
        test_seqs.append(last_seq)
        test_targets.append(last_item)


    train = pd.DataFrame({'user':train_users,'seq':train_seqs,'target':train_targets})

    test = pd.DataFrame({'user': test_users, 'seq': test_seqs, 'target': test_targets})

    if isSave:

        train.to_csv('train.csv', index=False)
        test.to_csv('test.csv', index=False)
        with open('info.pkl','wb+') as f:
            pickle.dump(user_all_items,f,pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_user_count,f,pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_item_count, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(user_map, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(item_map, f, pickle.HIGHEST_PROTOCOL)

    return train,test,\
           user_all_items,all_user_count,\
           all_item_count,user_map,item_map
```

```python id="ZGqCFhZXi40t" executionInfo={"status": "ok", "timestamp": 1637163535158, "user_tz": -330, "elapsed": 1249, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
if __name__ == '__main__':
    make_datasets('u.data', target_counts=3, seq_counts=5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="RYtor6PGjPVR" executionInfo={"status": "ok", "timestamp": 1637163542250, "user_tz": -330, "elapsed": 394, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="baa33e8b-0b09-4f91-bde8-20c4c8ff445a"
!head train.csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="lwns8w9BjSGc" executionInfo={"status": "ok", "timestamp": 1637163556180, "user_tz": -330, "elapsed": 566, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d510279c-5e59-45a7-fab6-9b56733b95c3"
!head test.csv
```

```python id="3YrieXGkjVew"

```
