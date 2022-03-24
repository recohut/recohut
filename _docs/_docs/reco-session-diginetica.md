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

```python id="qe7S4byMmAID"
# !mkdir -p /content/data
# !unzip -o /content/drive/MyDrive/TempData/Diginetica/diginetica.zip -d /content/data
# %cd /content/data
# !cp -r content/data/store/raw/* .
# !rm -r content
# !unzip -n dataset-train-diginetica.zip
# !rm dataset-train-diginetica.zip
# %cd /content
```

```python id="FDInDMqpsAxi" executionInfo={"status": "ok", "timestamp": 1621063637687, "user_tz": -330, "elapsed": 1153, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import csv
import tqdm
import datetime
import argparse
import numpy as np
import pandas as pd
import os
from collections import defaultdict

# for reproducibility
SEED = 666
np.random.seed(SEED)

# configuration parameters
dataset_path = '/content/data'
```

```python colab={"base_uri": "https://localhost:8080/"} id="WD3yd2s5wBsp" executionInfo={"status": "ok", "timestamp": 1621060898066, "user_tz": -330, "elapsed": 1205, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f9eec96a-b478-4740-ec9e-54ad5dbfffe1"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='train-item-views.csv', type=str)
parser.add_argument('--is_time_fraction', default=True, type=bool)  # split into different time fraction or not
parser.add_argument('--test_fraction', default='week', type=str)  # 'day' or 'week'
parser.add_argument('--threshold_sess', default=1, type=int)
parser.add_argument('--threshold_item', default=4, type=int)
args, unknown = parser.parse_known_args()

print('Start preprocess ' + args.dataset + ':')
```

```python id="HolgEkIlsDEZ" executionInfo={"status": "ok", "timestamp": 1621059714308, "user_tz": -330, "elapsed": 1086, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def generate_name_Id_map(name, map):
    """
    Given a name and map, return corresponding Id. If name not in map, generate a new Id.
    :param name: session or item name in dataset
    :param map: existing map, a dictionary: map[name]=Id
    :return: Id: allocated new Id of the corresponding name
    """
    if name in map:
        Id = map[name]
    else:
        Id = len(map.keys()) + 1
        map[name] = Id
    return Id
```

```python id="bRclhMwlsFJe" executionInfo={"status": "ok", "timestamp": 1621059730887, "user_tz": -330, "elapsed": 1259, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def read_data(dataset_path):
    """
    Load data from raw dataset.
    :param dataset_path: the full name of dataset including extension name
    :return sess_map: map from raw data session name to session Id, a dictionary sess_map[sess_name]=sessId
    :return item_map: map from raw data item name to item Id, a dictionary item_map[item_name]=itemId
    :return reformed_data: a list: each element is a action, which is a list of [sessId, itemId, time]
    """
    sess_map = {}
    item_map = {}
    reformed_data = []

    dataset_name = dataset_path.split('/')[-1]
    with open(dataset_path) as f:

        if dataset_name.split('-')[0] == 'train':
            # with sequence information
            reader = csv.DictReader(f, delimiter=';')
            timeframes = []
            for sample in reader:
                timeframes.append(int(sample['timeframe']))
            converter = 86400.00 / max(timeframes)
            f.seek(0)
            reader = csv.DictReader(f, delimiter=';')
            # load data
            for sample in tqdm.tqdm(reader, desc='Loading data'):
                sess = sample['sessionId']
                item = sample['itemId']
                date = sample['eventdate']
                timeframe = int(sample['timeframe'])
                if date:
                    time = int(datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()) + timeframe * converter
                else:
                    continue
                sessId = generate_name_Id_map(sess, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, time])
        else:
            print("Error: new csv data file!")

    # print raw dataset information
    print('Total number of sessions in dataset:', len(sess_map.keys()))
    print('Total number of items in dataset:', len(item_map.keys()))
    print('Total number of actions in dataset:', len(reformed_data))
    print('Average number of actions per user:', len(reformed_data) / len(sess_map.keys()))
    print('Average number of actions per item:', len(reformed_data) / len(item_map.keys()))

    return sess_map, item_map, reformed_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="pt_Wl3nvyIGW" executionInfo={"status": "ok", "timestamp": 1621063257919, "user_tz": -330, "elapsed": 2278, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8e3cd548-6829-434b-d5b4-b6c79fb3bfe4"
data = pd.read_csv(os.path.join(dataset_path, args.dataset), sep=';')
data.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ZSs6AdBayQFR" executionInfo={"status": "ok", "timestamp": 1621061358603, "user_tz": -330, "elapsed": 803, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="595e98f9-c6e9-4a11-b3eb-f0c979e0c1d5"
data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="YMnSJOdPwEI_" executionInfo={"status": "ok", "timestamp": 1621063311166, "user_tz": -330, "elapsed": 30652, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f7b8e2cf-d96b-469e-a0d8-f8fbcd8772eb"
sess_map, item_map, reformed_data = read_data(os.path.join(dataset_path, args.dataset))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="Na4xGMygxa3D" executionInfo={"status": "ok", "timestamp": 1621063315713, "user_tz": -330, "elapsed": 1690, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a9cecf7f-08c9-4e57-c41d-8143d411df6b"
display(sess_map['1'], item_map['81766'])
display(reformed_data[:10])
```

```python id="XfJBfdT3mtky" executionInfo={"status": "ok", "timestamp": 1621063124009, "user_tz": -330, "elapsed": 1841, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def generate_sess_end_map(sess_end, sessId, time):
    """
    Generate map recording the session end time.
    :param sess_end: the map recording session end time, a dictionary see_end[sessId]=end_time
    :param sessId:session Id of new action
    :param time:time of new action
    :return: sess_end: the map recording session end time, a dictionary see_end[sessId]=end_time
    """
    if sessId in sess_end:
        sess_end[sessId] = max(time, sess_end[sessId])
    else:
        sess_end[sessId] = time
    return sess_end
```

```python id="-CXY-lghsvty" executionInfo={"status": "ok", "timestamp": 1621063590209, "user_tz": -330, "elapsed": 1608, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def short_remove(reformed_data, args):
    """
    Remove data according to threshold
    :param reformed_data: loaded data, a list: each element is a action, which is a list of [sessId, itemId, time]
    :param args: args.threshold_item: minimum number of appearance time of item -1
                 args.threshold_sess: minimum length of session -1
    :return removed_data: result data after removing
    :return sess_end: a map recording session end time, a dictionary sess_end[sessId]=end_time
    """
    org_sess_end = dict()
    for [userId, _, time] in reformed_data:
        org_sess_end = generate_sess_end_map(org_sess_end, userId, time)

    # remove session whose length is 1
    sess_counter = defaultdict(lambda: 0)
    for [userId, _, _] in reformed_data:
        sess_counter[userId] += 1
    removed_data = list(filter(lambda x: sess_counter[x[0]] > 1, reformed_data))

    # remove item which appear less or equal to threshold_item
    item_counter = defaultdict(lambda: 0)
    for [_, itemId, _] in removed_data:
        item_counter[itemId] += 1
    removed_data = list(filter(lambda x: item_counter[x[1]] > args.threshold_item, removed_data))

    # remove session whose length less or equal to threshold_sess
    sess_counter = defaultdict(lambda: 0)
    for [userId, _, _] in removed_data:
        sess_counter[userId] += 1
    removed_data = list(filter(lambda x: sess_counter[x[0]] > args.threshold_sess, removed_data))

    # record session end time
    sess_end = dict()
    for [userId, _, time] in removed_data:
        sess_end = generate_sess_end_map(sess_end, userId, time)

    # print information of removed data
    print('Number of sessions after pre-processing:', len(set(map(lambda x: x[0], removed_data))))
    print('Number of items after pre-processing:', len(set(map(lambda x: x[1], removed_data))))
    print('Number of actions after pre-processing:', len(removed_data))
    print('Average number of actions per session:', len(removed_data) / len(set(map(lambda x: x[0], removed_data))))
    print('Average number of actions per item:', len(removed_data) / len(set(map(lambda x: x[1], removed_data))))

    return removed_data, sess_end
```

```python colab={"base_uri": "https://localhost:8080/"} id="r1o6IKWV56QT" executionInfo={"status": "ok", "timestamp": 1621063648351, "user_tz": -330, "elapsed": 4721, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c8454abe-a301-469b-d775-20686e10a67d"
# remove data according to occurrences time
removed_data, sess_end = short_remove(reformed_data, args)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} id="tb57NDIE7gQE" executionInfo={"status": "ok", "timestamp": 1621063863790, "user_tz": -330, "elapsed": 1211, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ca59c436-2ae7-440c-9004-33cf44a44669"
display(reformed_data[:10], removed_data[:10])
display(sess_end[1])
```

```python id="6kZLUEx4tDT3" executionInfo={"status": "ok", "timestamp": 1621063753984, "user_tz": -330, "elapsed": 2319, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def time_partition(removed_data, session_end, args):
    """
    Partition data according to time periods
    :param removed_data: input data, a list: each element is a action, which is a list of [sessId, itemId, time]
    :param session_end: a dictionary recording session end time, session_end[sessId]=end_time
    :param : args: args.test_fraction: time interval for each partition
    :return: time_fraction: a dictionary, the keys are different time periods, value is a list of actions in that
                            time period
    """
    if args.is_time_fraction:
        # split entire dataset by time interval
        time_fraction = dict()
        all_times = np.array(list(session_end.values()))
        max_time = max(all_times)
        min_time = min(all_times)

        if args.dataset == 'train-item-views.csv':
            # for DIGINETICA, choose the most recent 16 fraction and put left dataset in initial set
            if args.test_fraction == 'week':
                period_threshold = np.arange(max_time, min_time, -7 * 86400)
            elif args.test_fraction == 'day':
                period_threshold = np.arange(max_time, min_time, -86400)
            else:
                raise ValueError('invalid time fraction')
            period_threshold = np.sort(period_threshold)
            period_threshold = period_threshold[-17:]

        for [sessId, itemId, time] in removed_data:
            period = period_threshold.searchsorted(time) + 1
            # generate time period for dictionary keys
            if period not in time_fraction:
                time_fraction[period] = []
            # partition data according to period
            time_fraction[period].append([sessId, itemId, time])
    else:
        # if not partition, put all actions in the last period
        time_fraction = removed_data

    return time_fraction
```

```python id="TfTVTVsy7NEn" executionInfo={"status": "ok", "timestamp": 1621063920809, "user_tz": -330, "elapsed": 4086, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# partition data according to time periods
time_fraction = time_partition(removed_data, sess_end, args)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="43WFqmZT8GJ9" executionInfo={"status": "ok", "timestamp": 1621064305609, "user_tz": -330, "elapsed": 1801, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="942f780a-a1e9-4bc7-b400-93bd98fbaaca"
display(list(time_fraction.keys())[:10])
display(time_fraction[9][:10])
```

```python id="ZIJccCncvh9x" executionInfo={"status": "ok", "timestamp": 1621064452377, "user_tz": -330, "elapsed": 2886, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def generating_txt(time_fraction, sess_end, args):
    """
    Generate final txt file
    :param time_fraction: input data, a dictionary, the keys are different time periods,
                          value is a list of actions in that time period
    :param sess_end: session end time map, sess_map[sessId]=end_time
    :param : args: args.test_fraction: if not split, time interval for test partition
    """

    if args.is_time_fraction:
        # item map second time
        item_map = {}
        for period in sorted(time_fraction.keys()):
            time_fraction[period].sort(key=lambda x: sess_end[x[0]])
        for period in sorted(time_fraction.keys()):
            for i, [userId, itemId, time] in enumerate(time_fraction[period]):
                itemId = generate_name_Id_map(itemId, item_map)
                time_fraction[period][i] = [userId, itemId, time]

        # sort action according to time sequence
        for period in sorted(time_fraction.keys()):
            time_fraction[period].sort(key=lambda x: x[2])

        # generate text file
        for i, period in enumerate(sorted(time_fraction.keys())):
            with open('period_' + str(i) + '.txt', 'w') as file_train:
                for [userId, itemId, time] in time_fraction[period]:
                    file_train.write('%d %d\n' % (userId, itemId))
    else:
        # item map second time
        item_map = {}
        time_fraction.sort(key=lambda x: x[2])
        for i, [userId, itemId, time] in enumerate(time_fraction):
            itemId = generate_name_Id_map(itemId, item_map)
            time_fraction[i] = [userId, itemId, time]

        # sort action according to time sequence
        time_fraction.sort(key=lambda x: x[2])

        max_time = max(map(lambda x: x[2], time_fraction))
        if args.test_fraction == 'day':
            test_threshold = 86400
        elif args.test_fraction == 'week':
            test_threshold = 86400 * 7

        # generate text file
        item_set = set()
        with open('test.txt', 'w') as file_test, open('train.txt', 'w') as file_train:
            for [userId, itemId, time] in time_fraction:
                if sess_end[userId] < max_time - test_threshold:
                    file_train.write('%d %d\n' % (userId, itemId))
                    item_set.add(itemId)
                else:
                    file_test.write('%d %d\n' % (userId, itemId))
```

```python id="tYh09RCtmzU8" executionInfo={"status": "ok", "timestamp": 1621064463959, "user_tz": -330, "elapsed": 5547, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# generate final txt file
generating_txt(time_fraction, sess_end, args)
```

```python id="9gsiQOqO-J_-"

```
