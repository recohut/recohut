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

```python colab={"base_uri": "https://localhost:8080/"} id="ylELIzcYizTO" executionInfo={"status": "ok", "timestamp": 1637952090595, "user_tz": -330, "elapsed": 53483, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ee9eabe9-0a27-4047-a750-6453870902df"
!wget -q --show-progress https://paddlerec.bj.bcebos.com/datasets/Retailrocket/Retailrocket.zip
!mkdir raw && mv Retailrocket.zip raw
!cd raw && unzip Retailrocket.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="Z_yN6ki9jbB9" executionInfo={"status": "ok", "timestamp": 1637952169840, "user_tz": -330, "elapsed": 54562, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5f5fd955-9348-42a5-a9b3-c05effce8db4"
!cd raw && sort -k1 -n -t, events.csv > sorted_events.csv
!cd raw && sort -k1 -n -t, item_properties_part1.csv > sorted_item_properties_part1.csv
!cd raw && sort -k1 -n -t, item_properties_part2.csv > sorted_item_properties_part2.csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="QgoRi9KrokHz" executionInfo={"status": "ok", "timestamp": 1637953477021, "user_tz": -330, "elapsed": 656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ce602878-1658-4927-e5e6-ebd1ee140c66"
!head raw/sorted_events.csv
```

```python id="c1A7yWSAosqK" executionInfo={"status": "ok", "timestamp": 1637953494263, "user_tz": -330, "elapsed": 432, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a2d6769f-daf6-41b9-eb6d-9544ccd80e38" colab={"base_uri": "https://localhost:8080/"}
!head raw/sorted_item_properties_part1.csv
```

```python id="VXlmef5Xjq_A" executionInfo={"status": "ok", "timestamp": 1637952606489, "user_tz": -330, "elapsed": 443, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import csv
import os
import json
import gzip
import math
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
```

```python id="7-fVXR5akEuk" executionInfo={"status": "ok", "timestamp": 1637952608590, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# event fields
TIMESTAMP = 'timestamp'
PRESENTED_ITEMS = 'presentedItems'
EVENT_ITEM = 'clickedItem'
EVENT_TYPE = 'eventType'
EVENT_CONTEXT = 'context'
EVENT_USER_HASH = 'userHash'
EVENT_USER_ID = 'userId'
EVENT_SESSION_ID = 'sessionId'

# item fields
ITEM_ID = 'id'
```

```python id="z4u-6K6BlVKB" executionInfo={"status": "ok", "timestamp": 1637953105485, "user_tz": -330, "elapsed": 698, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
directory = '.'
directory_input = 'raw/'
input_path_events = directory_input + 'sorted_events.csv'
input_path_items = [
    directory_input + 'sorted_item_properties_part1.csv', directory_input + 'sorted_item_properties_part2.csv'
]
input_category_tree = directory_input + 'category_tree.csv'

delimiter = ','

datasets = 5
datasets_dir_prefix = './processed/data'
datasets_dirs = []
timestamp_first_event = 1430622004384
timestamp_last_event = 1442545187788


class RetailRocket:
    def __init__(self):
        self.items = dict()
        self.category_tree = dict()
        self.users_sessions = dict()
        self.next_session_id = 0
        self.items_in_datasets = dict()
        self.items_all_properties = set()
        self.items_mutable_properties = set()
        for i in range(datasets):
            self.items_in_datasets[i] = set()

    def prepare_items(self):
        self._read_category_tree()
        for input_path in input_path_items:
            self._add_items_properties(input_path)
        self._find_immutable_properties()

    def generate_events_file(self):
        rows = self._prepare_events()
        data = self._filter_events(rows)
        self._save_events_to_file(data)

    def save_items_to_file(self):
        print('Saving all items...')
        with gzip.open(f'{datasets_dir_prefix}/items.jsonl.gz', 'wt') as f:
            for item in self.items.values():
                f.write(item.transform_into_jsonl_format())
                f.write('\n')

        print('Saving splited items...')
        for i in range(datasets):
            items_set = self.items_in_datasets[i]
            with gzip.open(f'{datasets_dir_prefix}-{i+1}/items.jsonl.gz', 'wt') as f:
                for item_id in items_set:
                    item_jsonl = self.items[item_id].transform_into_jsonl_format()
                    f.write(item_jsonl)
                    f.write('\n')

    def _prepare_events(self):
        rows = []
        with open(input_path_events) as input_file:
            csv_reader = csv.reader(input_file, delimiter=delimiter)
            next(csv_reader, None)

            for line in csv_reader:
                event_jsonl = self._prepare_event_in_jsonl(line)
                if event_jsonl is not None:
                    ev_dict = json.loads(event_jsonl)
                    file_no = self.calculate_file_no(ev_dict['timestamp'])
                    row = [ev_dict['sessionId'], ev_dict['clickedItem'], ev_dict['timestamp'], event_jsonl, file_no]
                    rows.append(row)
        return rows

    def _filter_events(self, rows):
        columns = ['session_id', 'item_id', 'timestamp', 'event_jsonl', 'file_no']
        return self._filter_data(pd.DataFrame(rows, columns=columns))

    def _save_events_to_file(self, data):
        for i in range(datasets):
            d = f'{datasets_dir_prefix}-{i+1}'
            os.makedirs(d, exist_ok=True)
            datasets_dirs.append(d)

        os.makedirs(datasets_dir_prefix, exist_ok=True)
        datasets_dirs.append(datasets_dir_prefix)

        print('Saving all events dataset...')
        with gzip.open(f'{datasets_dir_prefix}/sessions.jsonl.gz', 'wt') as f:
            for _, row in data.iterrows():
                f.write(row['event_jsonl'] + '\n')

        print('Saving splited events datasets...')
        outputs = [gzip.open(f'{datasets_dir_prefix}-{i+1}/sessions.jsonl.gz', 'wt') for i in range(datasets)]
        for _, row in data.iterrows():
            if row['file_no'] < datasets:
                if row['item_id'] in self.items:
                    outputs[row['file_no']].write(row['event_jsonl'] + '\n')
                    self.items_in_datasets[row['file_no']].add(row['item_id'])
                else:
                    print(f'Item id: {row.item_id} is clicked but not in items dataset')
        map(lambda f: f.close(), outputs)

    def _add_items_properties(self, path):
        with open(path) as input_file:
            csv_reader = csv.reader(input_file, delimiter=delimiter)
            next(csv_reader, None)
            for line in csv_reader:
                self._add_item_property(line)

    def _add_item_property(self, line):
        assert len(line) == 4
        timestamp = int(line[0])
        item_id = line[1]
        property_name = line[2]
        value = line[3].strip().split(' ')
        if len(value) == 1:  # single value, no array is neccessary
            value = value[0]

        if item_id not in self.items.keys():
            self.items[item_id] = Item(item_id)

        self.items[item_id].add_property(property_name, timestamp, value)

        if property_name == "categoryid" and value in self.category_tree:
            category_path_ids = self._read_path_to_root(value)
            self.items[item_id].add_property("category_path_ids", timestamp, category_path_ids)

    def _read_path_to_root(self, leaf):
        current_node = leaf
        result = deque([current_node])

        while self.category_tree[current_node] != current_node:
            current_node = self.category_tree[current_node]
            result.appendleft(current_node)

        return result

    def _read_category_tree(self):
        with open(input_category_tree) as input_file:
            csv_reader = csv.reader(input_file, delimiter=delimiter)
            next(csv_reader, None)

            for line in csv_reader:
                if line[1] != "":
                    self.category_tree[int(line[0])] = int(line[1])
                else:  # when line describes root category
                    self.category_tree[int(line[0])] = int(line[0])

    def _find_immutable_properties(self):
        for item_id, item in self.items.items():
            for k, v in item.properties.items():  # k = property name, v = list of tuples (timestamp, value)
                self.items_all_properties.add(k)
                if len(v) > 1:  # if for all timestamps there is the same value => not muttable
                    for el in v:
                        if el[1] != v[0][1]:
                            self.items_mutable_properties.add(k)
                            break

        print(
            f'All items properties number: {len(self.items_all_properties)}, mutable: {len(self.items_mutable_properties)}'
        )
        for item_id, item in self.items.items():
            for k, v in item.properties.items():
                if k in self.items_mutable_properties:
                    item.mutable_properties[k] = v
                else:
                    item.immutable_properties[k] = v[0][1]  # take first value

    @staticmethod
    def normalize_context(r):
        d = dict()
        attribs = []
        for k, values in r.items():
            if not isinstance(values, list):
                values = [values]
            for v in values:
                if v.startswith('n'):  # number
                    f = float(v[1:])
                    if math.isinf(f):
                        print(f'Infinity! Bad value for {k} : {v}. Skipping...')
                        continue
                    d[k] = f
                else:
                    attribs.append(f'{k}|{v}')
        d['properties'] = attribs
        return d

    def _prepare_event_in_jsonl(self, line):
        def converter(o):
            if isinstance(o, datetime):
                return o.__str__()

        timestamp = int(line[0])
        user_id = int(line[1])
        item_id = line[3]

        if user_id not in self.users_sessions:
            self.users_sessions[user_id] = [timestamp, self.next_session_id]
            self.next_session_id += 1
        else:
            if timestamp - self.users_sessions[user_id][0] > 30 * 60 * 1000:  # 30 min * 60s * 1000ms
                self.users_sessions[user_id] = [timestamp, self.next_session_id]
                self.next_session_id += 1
            else:
                self.users_sessions[user_id][0] = timestamp  # update last activity in session

        if item_id in self.items:
            data = {
               TIMESTAMP: timestamp,
               EVENT_USER_ID: user_id,
               EVENT_TYPE: line[2],
               EVENT_ITEM: item_id,
               EVENT_SESSION_ID: self.users_sessions[user_id][1]
            }
            context = self._prepare_context(item_id, timestamp)
            if len(context) > 0:
                data[EVENT_CONTEXT] = RetailRocket.normalize_context(context)
            return json.dumps(data, default=converter, separators=(',', ':'))

    def _prepare_context(self, item_id, timestamp):
        context = {}
        for property, values in self.items[item_id].mutable_properties.items():
            ts, val = 0, 0
            for time, value in values:
                if timestamp >= time > ts:
                    ts = time
                    val = value
            if ts > 0:
                context[property] = val
        return context

    @staticmethod
    def _filter_data(data):  # based on 130L session-rec/preprocessing/preprocess_retailrocket.py

        session_lengths = data.groupby('session_id').size()
        data = data[np.in1d(data.session_id, session_lengths[session_lengths > 1].index)]

        item_supports = data.groupby('item_id').size()
        data = data[np.in1d(data.item_id, item_supports[item_supports >= 5].index)]

        session_lengths = data.groupby('session_id').size()
        data = data[np.in1d(data.session_id, session_lengths[session_lengths >= 2].index)]

        return data

    @staticmethod
    def calculate_file_no(ts):
        return int((ts - timestamp_first_event) / (1000 * 60 * 60 * 24 * 27))  # 1000ms * 60s * 60min * 24h * 27d


class Item:
    def __init__(self, id):
        self.id = str(id)
        self.properties = dict()  # all properties
        self.immutable_properties = dict()  # add to items.jsonl
        self.mutable_properties = dict()  # add to sessions.jsonl in context field

    def add_property(self, property, timestamp, value):
        if property not in self.properties.keys():
            self.properties[property] = list()
        self.properties[property].append((timestamp, value))

    def transform_into_jsonl_format(self):
        dt = {ITEM_ID: self.id}
        dt.update(RetailRocket.normalize_context(self.immutable_properties))
        return json.dumps(dt, separators=(',', ':'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="uFJnWWnplWVw" outputId="97d166e0-5eb4-434c-884e-c52808f06adc"
items = RetailRocket()
items.prepare_items()
items.generate_events_file()
items.save_items_to_file()
```

```python id="5tQ_wgIinOiz"

```
