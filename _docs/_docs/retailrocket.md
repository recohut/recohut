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

<!-- #region id="xg5m6EgrhS4t" -->
# RecSys RetailRocket
<!-- #endregion -->

<!-- #region id="fH3nvW9aCR6Q" -->
## Setup
<!-- #endregion -->

```python id="C8xVE9Qea_Jj"
# !pip install -q -U kaggle
# !pip install --upgrade --force-reinstall --no-deps kaggle
# !mkdir ~/.kaggle
# !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# # !kaggle datasets list
```

```python id="XcAYGpO6cmCE"
!kaggle datasets download -d retailrocket/ecommerce-dataset
!mkdir -p ./data && unzip ecommerce-dataset.zip
!mv ./*.csv ./data && rm ecommerce-dataset.zip
```

```python id="HxrmQPXZgKaB"
import os
import re
import time
import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

import bz2
import csv
import json
import operator

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```

<!-- #region id="3et4hlvIMnGu" -->
## Data Loading
<!-- #endregion -->

```python id="RZgv-uQyMBj0"
events_df = pd.read_csv('./data/events.csv')
category_tree_df = pd.read_csv('./data/category_tree.csv')
item_properties_1_df = pd.read_csv('./data/item_properties_part1.csv')
item_properties_2_df = pd.read_csv('./data/item_properties_part2.csv')
```

```python id="Nb6JTWWkcrZv"
item_prop_df = pd.concat([item_properties_1_df, item_properties_2_df])
item_prop_df.reset_index(drop=True, inplace=True)
del item_properties_1_df
del item_properties_2_df
```

```python id="kewR0WW1HoPI" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1611216398306, "user_tz": -330, "elapsed": 1498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="645140b1-0920-424e-97ae-a204c259251a"
events_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="kWMyH56HNVnC" executionInfo={"status": "ok", "timestamp": 1611216400383, "user_tz": -330, "elapsed": 1692, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c9bd9d2f-fdd3-4455-8f44-5272968c63b2"
item_prop_df.head()
```

<!-- #region id="VYBJSHOEOLC9" -->
- Property is the Item's attributes such as category id and availability while the rest are hashed for confidentiality purposes

- Value is the item's property value e.g. availability is 1 if there is stock and 0 otherwise

- Note: Values that start with "n" indicate that the value preceeding it is a number e.g. n277.200 is equal to `277.2`
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="yK69-KAAOX87" executionInfo={"status": "ok", "timestamp": 1611216653504, "user_tz": -330, "elapsed": 1573, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e75d8eb3-44de-4646-8b71-8658f793f9bd"
category_tree_df.head()
```

<!-- #region id="Pp0jvUKXA4GE" -->
## EDA
<!-- #endregion -->

<!-- #region id="SHJe-s8p6mXl" -->
Q: what are the items under category id `1016`?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="M-rO7aQSOh9t" executionInfo={"status": "ok", "timestamp": 1610435781867, "user_tz": -330, "elapsed": 3938, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="139f4ca4-6d63-4f63-d3cf-d053e6fd936c"
item_prop_df.loc[(item_prop_df.property == 'categoryid') & (item_prop_df.value == '1016')].sort_values('timestamp').head()
```

<!-- #region id="6AKenTUXDqhK" -->
Q: What is the parent category of `1016`?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SdfJDVpwDyFY" executionInfo={"status": "ok", "timestamp": 1611234517954, "user_tz": -330, "elapsed": 2112, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c464f4bf-57ff-4d6d-e61e-72ca8351ca76"
category_tree_df[category_tree_df.categoryid==1016]
```

<!-- #region id="Z_4-hu5pEGDu" -->
Q: What are items under category `213`?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1l8SbTWvD_Zc" executionInfo={"status": "ok", "timestamp": 1611234585996, "user_tz": -330, "elapsed": 6128, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="503e9b23-1185-46f4-8240-53acb0de14e5"
item_prop_df.loc[(item_prop_df.property == 'categoryid') & (item_prop_df.value == '213')].sort_values('timestamp').head()
```

<!-- #region id="SG8zLy9LEn0p" -->
visitors who bought something, assuming that there were no repeat users with different visitor IDs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GirnbnFQPnnf" executionInfo={"status": "ok", "timestamp": 1610433500322, "user_tz": -330, "elapsed": 1135, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="721b62cf-2272-484c-df8b-68ee7fbd957e"
customer_purchased = events_df[events_df.transactionid.notnull()].visitorid.unique()
all_customers = events_df.visitorid.unique()
customer_browsed = [x for x in all_customers if x not in customer_purchased]
print("%d out of %d"%(len(all_customers)-len(customer_browsed), len(all_customers)))
```

<!-- #region id="fQVOxoOlR1YP" -->
Snapshot of a random session with visitor id 102019
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 328} id="DrGTwmU_Rtw6" executionInfo={"status": "ok", "timestamp": 1610432898423, "user_tz": -330, "elapsed": 1395, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="726a5542-de68-433c-cbab-3eb1aa03222f"
events_df[events_df.visitorid == 102019].sort_values('timestamp')
```

```python colab={"base_uri": "https://localhost:8080/"} id="D3iV5U_FGRP8" executionInfo={"status": "ok", "timestamp": 1611218631523, "user_tz": -330, "elapsed": 1200, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="442555a6-5cb4-453f-d45d-e0c217eae1a9"
def _todatetime(dt):
  return datetime.datetime.fromtimestamp(int(dt/1000)).strftime('%Y-%m-%d %H:%M:%S')

print('Range of transaction dates = ', _todatetime(events_df['timestamp'].min()), 'to', _todatetime(events_df['timestamp'].max()))
```

<!-- #region id="dBx7rbofBRZu" -->
## Preprocessing
<!-- #endregion -->

```python id="hJ26GxG9EObt"
def preprocess_events(df):

  # convert unix time to pandas datetime
  df['date'] = pd.to_datetime(df['timestamp'], unit='ms', origin='unix')
  
  # label the events
  # events.event.replace(to_replace=dict(view=1, addtocart=2, transaction=3), inplace=True)

  # convert event to categorical
  df['event_type'] = df['event'].astype('category')

  # # drop the transcationid and timestamp columns
  # df.drop(['transactionid', 'timestamp'], axis=1, inplace=True)

  # # label encode
  # le_users = LabelEncoder()
  # le_items = LabelEncoder()
  # events['visitorid'] = le_users.fit_transform(events['visitorid'])
  # events['itemid'] = le_items.fit_transform(events['itemid'])
  
  # return train, valid, test
  return df
```

```python colab={"base_uri": "https://localhost:8080/"} id="i6B_f-huEjlj" executionInfo={"status": "ok", "timestamp": 1611219049016, "user_tz": -330, "elapsed": 1716, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d7a382a4-ceb4-4a44-a804-33a67cc7b0fa"
events_processed = preprocess_events(events_df)
events_processed.head()
```

```python id="bw2lm-vll8Bc"
dfx = events_df.sample(frac=0.01)
```

```python id="41ZS2fxEdkeb"
def sessionize(events_df: pd.DataFrame):

  session_duration = datetime.timedelta(minutes=30)
  gpby_visitorid = events_df.groupby('visitorid')

  session_list = []
  for a_visitorid in gpby_visitorid.groups:

    visitor_df = events_df.loc[gpby_visitorid.groups[a_visitorid], :].sort_values('date')
    if not visitor_df.empty:
        visitor_df.sort_values('date', inplace=True)

        # Initialise first session
        startdate = visitor_df.iloc[0, :]['date']
        visitorid = a_visitorid
        items_dict = dict([ (i, []) for i in events_df['event_type'].cat.categories ])
        for index, row in visitor_df.iterrows():

            # Check if current event date is within session duration
            if row['date'] - startdate <= session_duration:
            # Add itemid to the list according to event type (i.e. view, addtocart or transaction)
                items_dict[row['event']].append(row['itemid'])
                enddate = row['date']
            else:
                # Complete current session
                session_list.append([visitorid, startdate, enddate] + [ value for key, value in items_dict.items() ])
                # Start a new session
                startdate = row['date']
                items_dict = dict([ (i, []) for i in events_df['event_type'].cat.categories ])
                # Add current itemid
                items_dict[row['event']].append(row['itemid'])

        # If dict if not empty, add item data as last session.
        incomplete_session = False
        for key, value in items_dict.items():
            if value:
                incomplete_session = True
                break
        if incomplete_session:
            session_list.append([visitorid, startdate, enddate] + [value for key, value in items_dict.items()])

  return session_list
```

```python colab={"base_uri": "https://localhost:8080/"} id="78f7h51riXNa" executionInfo={"status": "ok", "timestamp": 1611219197499, "user_tz": -330, "elapsed": 71296, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="985652e7-1650-4af5-e4fb-a3a10c66e315"
session_list = sessionize(dfx)
sessions_df = pd.DataFrame(session_list, columns=['visitorid', 'startdate', 'enddate', 'addtocart', 'transaction', 'view'])
sessions_df.head()
```

```python id="DY8lnv5qMM35"
class BaseDataset(object):
    def __init__(self, input_path, output_path):
        super(BaseDataset, self).__init__()

        self.dataset_name = ''
        self.input_path = input_path
        self.output_path = output_path
        self.check_output_path()

        # input file
        self.inter_file = os.path.join(self.input_path, 'inters.dat')
        self.item_file = os.path.join(self.input_path, 'items.dat')
        self.user_file = os.path.join(self.input_path, 'users.dat')
        self.sep = '\t'

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # selected feature fields
        self.inter_fields = {}
        self.item_fields = {}
        self.user_fields = {}

    def check_output_path(self):
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

    def get_output_files(self):
        output_inter_file = os.path.join(self.output_path, self.dataset_name + '.inter')
        output_item_file = os.path.join(self.output_path, self.dataset_name + '.item')
        output_user_file = os.path.join(self.output_path, self.dataset_name + '.user')
        return output_inter_file, output_item_file, output_user_file

    def load_inter_data(self) -> pd.DataFrame():
        raise NotImplementedError

    def load_item_data(self) -> pd.DataFrame():
        raise NotImplementedError

    def load_user_data(self) -> pd.DataFrame():
        raise NotImplementedError

    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')

    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    def convert_user(self):
        try:
            input_user_data = self.load_user_data()
            self.convert(input_user_data, self.user_fields, self.output_user_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to user file\n')

    @staticmethod
    def convert(input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
            output_data[column] = input_data.iloc[:, column]
        with open(output_file, 'w') as fp:
            fp.write('\t'.join([selected_fields[column] for column in output_data.columns]) + '\n')
            for i in tqdm(range(output_data.shape[0])):
                fp.write('\t'.join([str(output_data.iloc[i, j])
                                    for j in range(output_data.shape[1])]) + '\n')

    def parse_json(self, data_path):
        with open(data_path, 'rb') as g:
            for l in g:
                yield eval(l)

    def getDF(self, data_path):
        i = 0
        df = {}
        for d in self.parse_json(data_path):
            df[i] = d
            i += 1
        data = pd.DataFrame.from_dict(df, orient='index')
        
        return data
```

```python id="_mqXrXmnMIlT"
class RETAILROCKETDataset(BaseDataset):
    def __init__(self, input_path, output_path, interaction_type, duplicate_removal):
        super(RETAILROCKETDataset, self).__init__(input_path, output_path)
        self.dataset_name = 'retailrocket'
        self.interaction_type = interaction_type
        assert self.interaction_type in ['view', 'addtocart',
                                         'transaction'], 'interaction_type must be in [view, addtocart, transaction]'
        self.duplicate_removal = duplicate_removal

        # input file
        self.inter_file = os.path.join(self.input_path, 'events.csv')
        self.item_file1 = os.path.join(self.input_path, 'item_properties_part1.csv')
        self.item_file2 = os.path.join(self.input_path, 'item_properties_part2.csv')
        self.sep = ','

        # output file
        if self.interaction_type == 'view':
            self.output_inter_file = os.path.join(self.output_path, 'retailrocket-view.inter')
        elif self.interaction_type == 'addtocart':
            self.output_inter_file = os.path.join(self.output_path, 'retailrocket-addtocart.inter')
        elif self.interaction_type == 'transaction':
            self.output_inter_file = os.path.join(self.output_path, 'retailrocket-transaction.inter')
        self.output_item_file = os.path.join(self.output_path, 'retailrocket.item')

        # selected feature fields
        if self.duplicate_removal:
            if self.interaction_type == 'view':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token',
                                     3: 'count:float'}
            elif self.interaction_type == 'addtocart':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token',
                                     3: 'count:float'}
            elif self.interaction_type == 'transaction':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token',
                                     3: 'count:float'}
        else:
            if self.interaction_type == 'view':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token'}
            elif self.interaction_type == 'addtocart':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token'}
            elif self.interaction_type == 'transaction':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token',
                                     3: 'transaction_id:token'}
        self.item_fields = {0: 'item_timestamp:float',
                            1: 'item_id:token',
                            2: 'property:token',
                            3: 'value:token_seq'}

    def convert_inter(self):
        if self.duplicate_removal:
            fin = open(self.inter_file, "r")
            fout = open(self.output_inter_file, "w")

            lines_count = 0
            for _ in fin:
                lines_count += 1
            fin.seek(0, 0)

            fout.write('\t'.join([self.inter_fields[column] for column in self.inter_fields.keys()]) + '\n')
            dic = {}

            for i in tqdm(range(lines_count)):
                if i == 0:
                    fin.readline()
                    continue
                line = fin.readline()
                line_list = line.split(',')
                key = (line_list[1], line_list[3])
                if line_list[2] == self.interaction_type:
                    if key not in dic:
                        dic[key] = (line_list[0], 1)
                    else:
                        if line_list[0] > dic[key][0]:
                            dic[key] = (line_list[0], dic[key][1] + 1)
                        else:
                            dic[key] = (dic[key][0], dic[key][1] + 1)

            for key in dic.keys():
                fout.write(dic[key][0] + '\t' + key[0] + '\t' + key[1] + '\t' + str(dic[key][1]) + '\n')

            fin.close()
            fout.close()
        else:
            fin = open(self.inter_file, "r")
            fout = open(self.output_inter_file, "w")

            lines_count = 0
            for _ in fin:
                lines_count += 1
            fin.seek(0, 0)

            fout.write('\t'.join([self.inter_fields[column] for column in self.inter_fields.keys()]) + '\n')

            for i in tqdm(range(lines_count)):
                if i == 0:
                    fin.readline()
                    continue
                line = fin.readline()
                line_list = line.split(',')
                if line_list[2] == self.interaction_type:
                    if self.interaction_type != 'transaction':
                        del line_list[4]
                    else:
                        line_list[4] = line_list[4].strip()
                    del line_list[2]
                    fout.write('\t'.join([str(line_list[i]) for i in range(len(line_list))]) + '\n')

            fin.close()
            fout.close()

    def convert_item(self):
        fin1 = open(self.item_file1, "r")
        fin2 = open(self.item_file2, "r")
        fout = open(self.output_item_file, "w")

        lines_count1 = 0
        for _ in fin1:
            lines_count1 += 1
        fin1.seek(0, 0)

        lines_count2 = 0
        for _ in fin2:
            lines_count2 += 1
        fin2.seek(0, 0)

        fout.write('\t'.join([self.item_fields[column] for column in self.item_fields.keys()]) + '\n')

        for i in tqdm(range(lines_count1)):
            if i == 0:
                line = fin1.readline()
                continue
            line = fin1.readline()
            line_list = line.split(',')
            fout.write('\t'.join([str(line_list[i]) for i in range(len(line_list))]))

        for i in tqdm(range(lines_count2)):
            if i == 0:
                line = fin2.readline()
                continue
            line = fin2.readline()
            line_list = line.split(',')
            fout.write('\t'.join([str(line_list[i]) for i in range(len(line_list))]))

        fin1.close()
        fin2.close()
        fout.close()
```

```python colab={"base_uri": "https://localhost:8080/"} id="HJOZgfHpNIHt" executionInfo={"status": "ok", "timestamp": 1611237234343, "user_tz": -330, "elapsed": 50991, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d325b981-696d-4f76-aaf8-8a295fdb419d"
# data_object = RETAILROCKETDataset('./data', '.', 'view', True)
# data_object.convert_inter()
# data_object.convert_item()
```

<!-- #region id="D_19B88xJufx" -->
## Feature Engineering
<!-- #endregion -->

<!-- #region id="iFBpIeaZJyr_" -->
Page Time
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="StuYb5igkAHR" executionInfo={"status": "ok", "timestamp": 1611220625580, "user_tz": -330, "elapsed": 2550, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="45de3b2c-9d93-4a81-bc90-2826c56e14aa"
sessions_df['pages'] = sessions_df['view'].apply(lambda x: len(x))
pages_more_than1 = sessions_df['pages'] > 1
pages_less_than1 = pages_more_than1.apply(lambda x: not x)
sessions_df.loc[pages_more_than1, 'pagetime'] = (sessions_df.loc[pages_more_than1, 'enddate'] - sessions_df.loc[pages_more_than1, 'startdate']) /\
                                                (sessions_df.loc[pages_more_than1, 'pages'] - 1)
sessions_df.loc[pages_less_than1, 'pagetime'] = pd.Timedelta(0)
sessions_df.head(10)
```

<!-- #region id="SEElLub9sQjg" -->
The rule of thumb on creating a simple yet effective recommender system is to downsample the data without losing quality. It means, you can take only maybe 50 latest transactions for each user and you still get the quality you want because behavior changes over-time.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="viVKSGKAsOex" executionInfo={"status": "ok", "timestamp": 1610439976574, "user_tz": -330, "elapsed": 2076, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4bbfd66e-613d-407e-a21b-6940c872159d"
trans = events_df[events_df['event'] == 'transaction']
trans2 = trans.groupby(['visitorid']).head(50)
trans2
```

```python id="bYPEhXJTtBQl"
visitors = trans['visitorid'].unique()
items = trans['itemid'].unique()

trans2['visitors'] = trans2['visitorid'].apply(lambda x : np.argwhere(visitors == x)[0][0])
trans2['items'] = trans2['itemid'].apply(lambda x : np.argwhere(items == x)[0][0])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="FxW7xi0YtYgt" executionInfo={"status": "ok", "timestamp": 1610440089282, "user_tz": -330, "elapsed": 1419, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a3861d50-ab77-4107-cf96-da7eaca5cbe9"
trans2.head()
```

<!-- #region id="8HmRqRC0tuFt" -->
Create the user-item matrix
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sMIQd7DBtg7i" executionInfo={"status": "ok", "timestamp": 1610440152639, "user_tz": -330, "elapsed": 11930, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="70845225-35c8-4360-d3ed-646543dc9081"
from scipy.sparse import csr_matrix

occurences = csr_matrix((visitors.shape[0], items.shape[0]), dtype='int8')

def set_occurences(visitor, item):
    occurences[visitor, item] += 1

trans2.apply(lambda row: set_occurences(row['visitors'], row['items']), axis=1)

occurences
```

<!-- #region id="dAfq2N60uBrf" -->
Co-occurrence is a better occurrence

Let’s construct an item-item matrix where each element means how many times both items bought together by a user. Call it the co-occurrence matrix.
<!-- #endregion -->

```python id="-eM9fETruJz3"
cooc = occurences.transpose().dot(occurences)
cooc.setdiag(0)
```

```python id="lJ7Nql5kEK8H"
  # split into train, test and valid
  train, test = train_test_split(events, train_size=0.9)
  train, valid = train_test_split(train, train_size=0.9)
  print('Train:{}, Valid:{}, Test:{}'.format(train.shape,
                                            valid.shape,
                                            test.shape))
```

<!-- #region id="C1jTXQCRQJKO" -->
https://nbviewer.jupyter.org/github/tkokkeng/EB5202-RetailRocket/blob/master/retailrocket-features.ipynb
<!-- #endregion -->

<!-- #region id="Q1dfT0E_QHB8" -->
<!-- #endregion -->

<!-- #region id="Ms964u54imRX" -->
## Matrix factorization model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HK5IyEuTiagt" executionInfo={"status": "ok", "timestamp": 1609587143019, "user_tz": -330, "elapsed": 1265, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5692ab74-c1fc-47e7-f3ad-e8a94bc22035"
# store the number of visitors and items in a variable
n_users = events.visitorid.nunique()
n_items = events.itemid.nunique()

# set the number of latent factors
n_latent_factors = 5

# import the required layers
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Flatten

# create input layer for items
item_input = Input(shape=[1],name='Items')

# create embedding layer for items
item_embed = Embedding(n_items,
                       n_latent_factors,
                       name='ItemsEmbedding')(item_input)
item_vec = Flatten(name='ItemsFlatten')(item_embed)

# create the input and embedding layer for users also
user_input = Input(shape=[1],name='Users')
user_embed = Embedding(n_users,
                       n_latent_factors, 
                       name='UsersEmbedding')(user_input)
user_vec = Flatten(name='UsersFlatten')(user_embed)

# create a layer for the dot product of both vector space representations
dot_prod = keras.layers.dot([item_vec, user_vec],axes=[1,1],
                             name='DotProduct')

# build and compile the model
model = keras.Model([item_input, user_input], dot_prod)
model.compile('adam', 'mse')
model.summary()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 422} id="yuNltIrJn_Fa" executionInfo={"status": "ok", "timestamp": 1609586509274, "user_tz": -330, "elapsed": 1502, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="45d1a810-1b16-48f6-de87-366995d334ec"
keras.utils.plot_model(model, 
                       to_file='model.png', 
                       show_shapes=True, 
                       show_layer_names=True)
from IPython import display
display.display(display.Image('model.png'))
```

```python id="YzbLeB9OoNdd"
# train and evaluate the model
model.fit([train.visitorid.values, train.itemid.values], train.event.values, epochs=50)
score = model.evaluate([test.visitorid, test.itemid], test.event)
print('mean squared error:', score)
```

<!-- #region id="7T9jBk4m4oiV" -->
## Neural net model
<!-- #endregion -->

```python id="OcBOy0QcoXeV"
n_lf_visitor = 5
n_lf_item = 5

item_input = Input(shape=[1],name='Items')
item_embed = Embedding(n_items + 1,
                           n_lf_visitor, 
                           name='ItemsEmbedding')(item_input)
item_vec = Flatten(name='ItemsFlatten')(item_embed)

visitor_input = Input(shape=[1],name='Visitors')
visitor_embed = Embedding(n_visitors + 1, 
                              n_lf_item,
                              name='VisitorsEmbedding')(visitor_input)
visitor_vec = Flatten(name='VisitorsFlatten')(visitor_embed)

concat = keras.layers.concatenate([item_vec, visitor_vec], name='Concat')
fc_1 = Dense(80,name='FC-1')(concat)
fc_2 = Dense(40,name='FC-2')(fc_1)
fc_3 = Dense(20,name='FC-3', activation='relu')(fc_2)

output = Dense(1, activation='relu',name='Output')(fc_3)

optimizer = keras.optimizers.Adam(lr=0.001)
model = keras.Model([item_input, visitor_input], output)
model.compile(optimizer=optimizer,loss= 'mse')

model.fit([train.visitorid, train.itemid], train.event, epochs=50)
score = model.evaluate([test.visitorid, test.itemid], test.event)
print('mean squared error:', score)
```

<!-- #region id="XwiMEOwi55mz" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="CrTjO7i46Htd" executionInfo={"status": "ok", "timestamp": 1609588005823, "user_tz": -330, "elapsed": 58544, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="42de1144-29fd-4c04-e698-ae3b9fb1352b"
user_activity_count = dict()
for row in events.itertuples():
    if row.visitorid not in user_activity_count:
        user_activity_count[row.visitorid] = {'view':0 , 'addtocart':0, 'transaction':0};
    if row.event == 'addtocart':
        user_activity_count[row.visitorid]['addtocart'] += 1 
    elif row.event == 'transaction':
        user_activity_count[row.visitorid]['transaction'] += 1
    elif row.event == 'view':
        user_activity_count[row.visitorid]['view'] += 1 

d = pd.DataFrame(user_activity_count)
dataframe = d.transpose()

# Activity range
dataframe['activity'] = dataframe['view'] + dataframe['addtocart'] + dataframe['transaction']

# removing users with only a single view
cleaned_data = dataframe[dataframe['activity']!=1]

cleaned_data.head()
```

<!-- #region id="X-lA1bAG7Fpq" -->
Since the data is very sparse, data cleaning is required to reduce the inherent noise. Steps performed

- Found activity per item basis. Activity is view / addtocart / transaction
- Removed items with just a single view/activity (confirmed that, addtocard ones have both view+addtocart)
- Removed users with no activity
- Gave new itemId and userId to all users and items with some event attached and not removed in above steps.
<!-- #endregion -->

<!-- #region id="Z5lDAhMl7rPw" -->
---
<!-- #endregion -->
