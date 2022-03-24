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

<!-- #region id="46tlPHAP1laa" -->
# IEEE BigData 2021 RecSys Dataset Wrangling and EDA
<!-- #endregion -->

<!-- #region id="ox1UmaUg1qyD" -->
## Setup
<!-- #endregion -->

<!-- #region id="tc5woWat-jKh" -->
### Imports
<!-- #endregion -->

```python id="PEa46Hco-Qpv" executionInfo={"status": "ok", "timestamp": 1636041841007, "user_tz": -330, "elapsed": 654, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from pathlib import Path
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import os
from os import path as osp
import logging
import sys
from datetime import datetime
from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
%matplotlib inline
```

<!-- #region id="dJSpp5mX-QoK" -->
### Data Ingestion
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7ReVgLn9-njI" executionInfo={"status": "ok", "timestamp": 1636029927948, "user_tz": -330, "elapsed": 15468, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="721527e4-b364-41ca-e2b1-62dad4e34a65"
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/train.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/item_info.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track1_testset.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track2_testset.parquet.snappy
```

<!-- #region id="7Ps4TMbC_k2c" -->
### Params
<!-- #endregion -->

```python id="46U15v5T_mD1" executionInfo={"status": "ok", "timestamp": 1636038989828, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Args:
    datapath_bronze = '/content'
    datapath_silver = '/content'

    filename_trainset = 'train.parquet.snappy'
    filename_items_metadata = 'df_items_metadata.parquet.snappy'
    filename_items_timestamp = 'df_items_timestamp.parquet.snappy'
```

```python id="Uv-KojLCJ_gV" executionInfo={"status": "ok", "timestamp": 1636038989832, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args()
```

<!-- #region id="s05VzKggClu2" -->
### Logging
<!-- #endregion -->

```python id="GiTSITMLCmrt" executionInfo={"status": "ok", "timestamp": 1636038990797, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
logging.basicConfig(stream=sys.stdout,
                    level = logging.INFO,
                    format='%(asctime)s [%(levelname)s] : %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('IEEE21 Logger')
```

<!-- #region id="r1oREl6P-QmN" -->
## Utilities
<!-- #endregion -->

```python id="y0JYUoiLfDX8" executionInfo={"status": "ok", "timestamp": 1636039455333, "user_tz": -330, "elapsed": 910, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def read_data(name):
    """Read data files from the given path/dataset name"""
    name_path_dict = {
        'trainset': osp.join(args.datapath_bronze,args.filename_trainset),
        'items_ts': osp.join(args.datapath_silver,args.filename_items_timestamp),
        'items_meta': osp.join(args.datapath_silver,args.filename_items_metadata),
    }

    return pd.read_parquet(name_path_dict[name])
```

```python id="WOftjONWg0PG" executionInfo={"status": "ok", "timestamp": 1636039365099, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def extract_items_timestamp():
    """Extract item id and timestamps in a dataframe"""
    file_savepath = osp.join(args.datapath_silver,args.filename_items_timestamp)

    if not osp.exists(file_savepath):
        logger.info('Extracting Items timestamps')
        df = read_data('trainset')
        item_purchase_history = [history.split(',') for history in df.user_click_history.values]
        item_purchase_history = [item for user_history in item_purchase_history for item in user_history]
        df = pd.DataFrame()
        df[['item_id','timestamp']] = [item.split(':') for item in item_purchase_history]

        df.to_parquet(file_savepath, compression='snappy')
        logger.info('Extraction finished. File saved at {}'.format(file_savepath))
    else:
        logger.info('File already exists at {}'.format(file_savepath))
```

```python id="WXrFaTyy-Qiw" executionInfo={"status": "ok", "timestamp": 1636039400525, "user_tz": -330, "elapsed": 593, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def extract_items_metadata():
    """Extract item purchase details in a dataframe"""
    file_savepath = osp.join(args.datapath_silver,'df_item_metadata.parquet.snappy')

    if not osp.exists(file_savepath):
        logger.info('Extracting Items metadata')
        df = read_data('items_ts')
        
        item_metadata = []
        with tqdm(total=len(df.item_id.unique())) as pbar:
            for item_id in df.item_id.unique():
                item_timestamps = df[df.item_id == item_id].timestamp
                item_metadata.append({
                    'item_id': item_id,
                    'purchase_total': len(item_timestamps),
                    'purchase_start': item_timestamps.min(),
                    'purchase_end': item_timestamps.max()
                })
                pbar.update()

        df_item_metadata = pd.DataFrame(item_metadata)
        for column in ['purchase_total', 'purchase_start', 'purchase_end']:
            df_item_metadata[column] = pd.to_numeric(df_item_metadata[column])
        df_item_metadata = df_item_metadata.sort_values(by='purchase_start', ascending=True)
        df_item_metadata['purchase_duration'] = df_item_metadata['purchase_end'] - df_item_metadata['purchase_start']
        df_item_metadata.to_parquet(file_savepath, compression='snappy')
        logger.info('Extraction finished. File saved at {}'.format(file_savepath))
    else:
        logger.info('File already exists at {}'.format(file_savepath))
```

```python id="3Sn2JIRR-SxZ" executionInfo={"status": "ok", "timestamp": 1636041877527, "user_tz": -330, "elapsed": 658, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def change_into_datetime(time_int):
    """
    input: 1582992009 (int)
    output: 2020-02-29 16:00:09 (str)
    """
    return datetime.utcfromtimestamp(time_int).strftime('%Y-%m-%d %H:%M:%S')
```

```python id="2kFRqp28swTA" executionInfo={"status": "ok", "timestamp": 1636042235333, "user_tz": -330, "elapsed": 725, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def change_date_format(df):
    """change datetime into human-readable format"""
    logger.info('Changing datetime format in trainset')
    t = int(df.at[i, 'time'])
    df.at[i, 'datetime'] = change_into_datetime(t)

    # change time in user_click_history into human readable format
    # notice that some user_click_history == '0:0', which means there is no click history for this user
    t = df.at[i, 'user_click_history']
    if t == '0:0':
        df.at[i, 'user_click_history'] = ''
    else:
        new_user_click_history = [sample.split(':')[0] + ':' + change_into_datetime(int(sample.split(':')[1])) for sample in t.split(',')]
        df.at[i, 'user_click_history'] = ','.join(new_user_click_history)
    return df
```

<!-- #region id="Wxxp2lEJ_cis" -->
## Data Loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="a0zLQ9JG_d_T" executionInfo={"status": "ok", "timestamp": 1636039458663, "user_tz": -330, "elapsed": 2447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8d8fe1f0-10d5-4a13-d6a5-12f25598f5d6"
df = read_data('trainset')
df.head()
```

<!-- #region id="Fxu7XOj-_LFj" -->
## Data Processing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="b8WfStDdjRyG" executionInfo={"status": "ok", "timestamp": 1636039561503, "user_tz": -330, "elapsed": 62401, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f76119d1-9a0b-4cec-bdd9-74ccf88dbd13"
logger.info('JOB START: ITEM_TIMESTAMP_EXTRACTION')
extract_items_timestamp()
logger.info('JOB END: ITEM_TIMESTAMP_EXTRACTION')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 134, "referenced_widgets": ["baae497af03b478f8c420185c1230dbc", "b2a266e5941e4b00ba202d21f5be3215", "151b0385bef94cf9a1c56afd0628eb4d", "8e6ff5db194e4168b4205c42c1758eb1", "381555894f3b4074bfe1b944f3745743", "a943c0f8142e45bc9eacb05fab727428", "796de2699ec14756b871d50678c5c31d", "8f30626dee284bad96cf1d7732243a49", "138ef5d06a1341a385d2d3a50d279e09", "47caabbca961475fbce7eff25453b3d8", "08b17df019c34fe88d4de8ddbd346be3"]} id="igXYmBc5Bic7" executionInfo={"status": "ok", "timestamp": 1636035321701, "user_tz": -330, "elapsed": 281735, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="32a337b4-c07d-4e83-accd-2772d4a79f7e"
logger.info('JOB START: ITEM_META_EXTRACTION')
extract_items_metadata()
logger.info('JOB END: ITEM_META_EXTRACTION')
```

```python id="FwBhWo5at3kK" executionInfo={"status": "ok", "timestamp": 1636042360862, "user_tz": -330, "elapsed": 1610, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
logger.info('JOB START: TRAINSET_DATETIME_FORMATING')
df = read_data('trainset')
df_new = change_date_format(df)
logger.info('JOB END: TRAINSET_DATETIME_FORMATING')
```

<!-- #region id="_RzoDXH3-Sv6" -->
## EDA
<!-- #endregion -->

<!-- #region id="PcBIMp6A-StM" -->
### Items Sales Chronology
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="YzAUgRg4lxEF" executionInfo={"status": "ok", "timestamp": 1636040136805, "user_tz": -330, "elapsed": 4790, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="91360395-75fe-42f2-adfe-398cc85dded6"
df_items_timestamp = read_data('items_ts')
df_items_timestamp.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 325} id="PhXXrLnK-duL" executionInfo={"status": "ok", "timestamp": 1636040299284, "user_tz": -330, "elapsed": 869, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7e5dbf1a-c057-491f-9be2-4e067928900e"
logger.disabled = True
plt.figure(figsize=(30,10))
ax = plt.subplot()
ax.axes.xaxis.set_visible(False)
plt.xticks([])
plt.setp(ax.get_xticklabels(), visible=False)

count = 0
for item_id in df_items_timestamp.item_id.unique():
    if item_id != 0:
        item_timestamps = df_items_timestamp[df_items_timestamp.item_id == item_id].timestamp
        plt.hist(item_timestamps.values, 100, alpha=0.5, label=item_id)
        if count > 20:
            break
        count = count + 1
plt.show()
plt.close()
```

<!-- #region id="-gbBwqmQlsO-" -->
### Items Sales Durations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="jErkq81c-dxu" executionInfo={"status": "ok", "timestamp": 1636040313460, "user_tz": -330, "elapsed": 645, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c156dbc9-96a4-40da-b48e-73e417fbc9a3"
df_items_metadata = read_data('items_meta')
df_items_metadata.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 349} id="nhY1FQEZ-drE" executionInfo={"status": "ok", "timestamp": 1636040316207, "user_tz": -330, "elapsed": 897, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bc67af5b-90b2-429b-8bde-97656381569f"
count = 0
x_data = []
x_error = []
y_data = []
for index, item in df_items_metadata.iterrows():
    if item.item_id is not '0':
        y_data.append(item.purchase_end - item.purchase_start)
        x_data.append(item.item_id)
        x_error.append(item.purchase_start)
        if count > 20:
            break
        count = count + 1
y_pos = np.arange(len(x_data))
        
plt.figure(figsize=(30,10))
ax = plt.subplot()

ax.barh(y_pos, y_data, left=x_error, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(x_data)
ax.invert_yaxis()
ax.set_xlabel('Timestamp')
ax.set_title('Item Sale Duration')

plt.show()
plt.close()
```

<!-- #region id="lIxhtO7TgZ7_" -->
### Frequency Histograms of Clicked and Purchased Items
<!-- #endregion -->

```python id="x0JZTIY4njy5"
N_ITEMS = 380

clickFreq = [0]*N_ITEMS  # frequency array for clicking
purchFreq = [0]*N_ITEMS  # frequency array for purchase

# read data to pd dataframe
trainSet = read_data('trainset')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 313, "referenced_widgets": ["04244fa527204ad5a4f5ec0e58acb406", "16f1c1924119453a8ae65c81ad6fd835", "5917b4dd29e34e2da615921069a7535f", "6c69bcc9dada4e26b93ab119804a5413", "2c8f545f4cf241a6b7ce54b10d3ec70a", "019f5437ca084e5c81ff8d6693b1dca3", "63111a1a363e4a84b1395f9c3bfebdd6", "a3ebef4532a2487482e53e463fa97be3", "f5bcaf5878144fc2b6d9f6e7bf06474c", "628e1c35da164460b6e5542f4f3c836a", "feda88442dd64a09a221739745ba8d37"]} id="Oj3qJKkynlLG" executionInfo={"status": "ok", "timestamp": 1636040639818, "user_tz": -330, "elapsed": 9432, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7c25c212-9f41-47f8-a381-548fec08d5e0"
# iterate thru data and parse items clicked
# parse by comma first, then delete everthing after colon
for line in tqdm(trainSet.user_click_history):
    clickSeries = line.split(',')
    itemList = [item.partition(':')[0] for item in clickSeries]
    for item in itemList:
        # increment frequency
        item = int(item)
        clickFreq[item] = clickFreq[item] + 1

# sort the frequency arrays decreasing
# clickFreq.sort(reverse = True)

# draw the two histograms
plt.bar(list(range(N_ITEMS)), clickFreq)
plt.ylabel("Frequency")
plt.title("Click Frequency")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 313, "referenced_widgets": ["84cd1f671ccb43d1903ba8f6159a3a48", "5253f6f1ed7a46589ac1ed90014fd45d", "5a9986bddfd4435aac12202f921ffdf7", "613150c4038147edabdc3c098285f9c5", "da41ff13591a432786300a4ac8b0fe35", "bcd9d95841af444da8df799f88dd09cd", "f817daefe1b44a88aa87e2c43abce27c", "e9fd20cdafe848349d4adae2ab1bca5b", "1570a70fc0e541daa704d6fabe760aeb", "19ece323c8254ed39a7182cad2232f07", "98149c31643643ca8bd9f071710e4b0d"]} id="squOgTagnHPV" executionInfo={"status": "ok", "timestamp": 1636040663606, "user_tz": -330, "elapsed": 9081, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2da54fa6-19fd-44a6-e9dd-c8a986d7c11e"
# iterate thru data and parse items purchased
# parse items exposed by comma, then check if purchased in column 'labels'
for r in tqdm(range(trainSet.shape[0])):
    temp = trainSet.exposed_items[r]
    exposedItems = temp.split(',')
    temp = trainSet.labels[r]
    labels = temp.split(',')
    # increment purchase frequency if label==1
    for i in range(len(exposedItems)):
        if int(labels[i])==1:
            purchFreq[int(exposedItems[i])] = purchFreq[int(exposedItems[i])] + 1

# sort the frequency arrays decreasing
# purchFreq.sort(reverse = True)

plt.bar(list(range(N_ITEMS)), purchFreq)
plt.ylabel("Frequency")
plt.title("Purchase Frequency")
plt.show()
```

<!-- #region id="5QR22-gJnbgW" -->
### User Click History Analysis
<!-- #endregion -->

<!-- #region id="LOZQ-1qyu1fK" -->
**calculating len of user_click_history**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 328, "referenced_widgets": ["cf9b32a7b01b42259557e7d641b1e98d", "d717221e2ab245c1963336eceef699a6", "2d60efd664604c60a76ce437e4301045", "725657621bf04aa5b7e7660081255927", "51bb6fb107014b4981e460ee7f1447a3", "e72481170dee40c7ab6849c43918ed68", "7e5aa1749beb4697b00cbafb597cb8bd", "60530a193e1b455fa72169c24e165822", "55ee5c7308fa4a538a5ec25e3cc19ff4", "4980dcf2ae3d4ef3bc72edb3e580a32a", "3f58e3cc0db74b94ab56e45143647630"]} id="CbSjLMnOuiKA" executionInfo={"status": "ok", "timestamp": 1636042461745, "user_tz": -330, "elapsed": 5837, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3813129c-29aa-4a83-d2ba-6a691cef2a21"
for i in tqdm(range(df_new.shape[0])):
    user_click_history = df_new.at[i, 'user_click_history']
    if user_click_history == '':
        len_user_click_history = 0
    else:
        len_user_click_history = int(len(user_click_history.split(',')))
    df_new.at[i, 'len_user_click_history'] = len_user_click_history

arr = np.array(df_new['len_user_click_history'])
plt.hist(arr, color='blue', edgecolor='black', bins=100)
plt.title('Histogram of len_user_click_history')
plt.xlabel('# of user_click_history')
plt.ylabel('Count')
plt.show()
```

<!-- #region id="Zpst3YOmurbr" -->
**what items are users clicking?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["16da091c454148739577d0eb4d8ae93d", "7a5c5778c2d845c3b129bd4afec86f6c", "bcbb681bede147b1b884182f7c616192", "78378698f4d847f6880b2439b84c832f", "570ae12eaef747eb950ba0d6cc3ba0e1", "9b8ad818ec0041c5ba55dbfe76f183f6", "9b2d9e2fcad6494aa8a9d6d56b1f5d5b", "26325e82063c4f6c88fe1b7149796b31", "17393e0a5d484d058d115be01d1c59fb", "cda1128b9f5049b08a863bb8cf39726d", "b598c084191c4ee39e2e4f5db9c05810"]} id="u96dH7EQu94X" executionInfo={"status": "ok", "timestamp": 1636042628421, "user_tz": -330, "elapsed": 9003, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ad2eccbd-afda-490a-ff58-b4fc2e24e770"
counter = defaultdict(lambda: 0)
for i in tqdm(range(df_new.shape[0])):
    user_click_history = df_new.at[i, 'user_click_history']
    if user_click_history == '':
        continue
    for c in user_click_history.split(','):
        itemid = c.split(':')[0]
        counter[itemid] += 1
counter = dict(counter)
counter_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="64cH46T-vPSE" executionInfo={"status": "ok", "timestamp": 1636042636648, "user_tz": -330, "elapsed": 5375, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4093d88b-9198-430f-db99-acf1e17980e3"
nums_to_plot = counter_sorted
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item clicked? (All)')
plt.xlabel('itemid')
plt.ylabel('item clicked times')
plt.tight_layout()
plt.show()


nums_to_plot = counter_sorted[:40]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item clicked? (Top)')
plt.xlabel('itemid')
plt.ylabel('item clicked times')
plt.tight_layout()
plt.show()


nums_to_plot = counter_sorted[-40:]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item clicked? (Bottom)')
plt.xlabel('itemid')
plt.ylabel('item clicked times')
plt.tight_layout()
plt.show()
```

<!-- #region id="xr2iM733vQzX" -->
### Clicks by Session
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["fc1dd03886ab4172b358d76c47d36167", "97b5fff4e176405d865c9d5d162a2d76", "0f7b24daa2e14b21ab25a277a6c564df", "39d36da480274deea228a35fca197566", "aa90e9e68108484885e94afcd167eaa1", "35e877509ea144068ccccce063019885", "f5f07ea91d38417f8ef53b4cd5c2a4ff", "d21155e280984bbdb89594d11f3ee942", "952ff13e804145b4992a9b8c5ee289b7", "bcff1eeb4ef54ab59634b4476b536cd4", "d572c25fdaa14a028fb0cb9612cda904"]} id="KMvyegr_vg1p" executionInfo={"status": "ok", "timestamp": 1636042745772, "user_tz": -330, "elapsed": 8263, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="02b1c35d-6aeb-4785-b058-cb614fc2a68e"
counter = defaultdict(lambda: 0)
for i in tqdm(range(df_new.shape[0])):
    user_click_history = df_new.at[i, 'user_click_history']
    if user_click_history == '':
        continue
    for c in user_click_history.split(','):
        itemid = c.split(':')[0]
        counter[itemid] += 1
counter = dict(counter)
counter_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="eOdxA0OxvmZs" executionInfo={"status": "ok", "timestamp": 1636042757560, "user_tz": -330, "elapsed": 2932, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="01cc9b30-c1e1-476a-e742-7d27383cec83"
counter_sorted_session1 = [s for s in counter_sorted if 1 <= int(s[0]) <= 39]
counter_sorted_session2 = [s for s in counter_sorted if 40 <= int(s[0]) <= 147]
counter_sorted_session3 = [s for s in counter_sorted if 148 <= int(s[0]) <= 381]

nums_to_plot = counter_sorted_session1
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item clicked? (Session 1)')
plt.xlabel('itemid')
plt.ylabel('item clicked times')
plt.tight_layout()
plt.show()

nums_to_plot = counter_sorted_session2[:40]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item clicked? (Session 2)')
plt.xlabel('itemid')
plt.ylabel('item clicked times')
plt.tight_layout()
plt.show()

nums_to_plot = counter_sorted_session3[:40]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item clicked? (Session 3)')
plt.xlabel('itemid')
plt.ylabel('item clicked times')
plt.tight_layout()
plt.show()
```

<!-- #region id="VVE2t0F0v0bG" -->
### Expose by Session
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["6d49809e60a44886b77778d362b5cb02", "0d7e54f488a94ae4b6d91ce8dda14edb", "4a2d24ddb5454fd7a5110dc4ddcc2a1a", "684a7aa7f2454bebbce4009d12dd45d0", "cb54437b4d8d4933ab978c63191d2b3a", "6ec4dbc83bfe4abc8b2b1d74ce5f1366", "b75fbf0c1fb44c55baf0981e7145787b", "1167fb710abb473a9827791344c1ec84", "338254b4829d410687ea21d0832794b8", "bade0660d390499b83701091c82523de", "d389b26d00224d26ae982408b2e06407"]} id="wAOuGdEpwEpq" executionInfo={"status": "ok", "timestamp": 1636042859654, "user_tz": -330, "elapsed": 5988, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="499362a7-c664-44e4-f5ec-d462b453f480"
## expose should only be counted when users are in that session
# s1: 0~2, s2: 3~5, s3: 6~9

counter = defaultdict(lambda: 0)
for i in tqdm(range(df_new.shape[0])):
    user_exposed_history = df_new.at[i, 'exposed_items']
    num_bought = sum([int(l) for l in df_new.at[i, 'labels'].split(',')])
    for idx, itemid in enumerate(user_exposed_history.split(',')):
        if 0 <= idx <= 2:
            counter[itemid] += 1
        elif 3 <= idx <= 5 and num_bought >= 3: 
            counter[itemid] += 1
        elif 6 <= idx <= 8 and num_bought >= 6:
            counter[itemid] += 1
counter = dict(counter)
counter_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="9gLnoMxxwJ1B" executionInfo={"status": "ok", "timestamp": 1636042862258, "user_tz": -330, "elapsed": 2611, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="731c04fd-4bdb-4e23-e6fc-fafa07de5153"
counter_sorted_session1 = [s for s in counter_sorted if 1 <= int(s[0]) <= 39]
counter_sorted_session2 = [s for s in counter_sorted if 40 <= int(s[0]) <= 147]
counter_sorted_session3 = [s for s in counter_sorted if 148 <= int(s[0]) <= 381]

nums_to_plot = counter_sorted_session1
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item exposed? (Session 1)')
plt.xlabel('itemid')
plt.ylabel('item exposed times')
plt.tight_layout()
plt.show()

nums_to_plot = counter_sorted_session2[:40]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item exposed? (Session 2)')
plt.xlabel('itemid')
plt.ylabel('item exposed times')
plt.tight_layout()
plt.show()

nums_to_plot = counter_sorted_session3[:40]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item exposed? (Session 3)')
plt.xlabel('itemid')
plt.ylabel('item exposed times')
plt.tight_layout()
plt.show()
```

<!-- #region id="l80yCkwcwN5t" -->
### Bought by Session
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["465201df928848a48bb2de8944e4c174", "4c6112b87b6b4c5abbb01050bb95a9d2", "cba77e688d4a4bf6b781675d319ddf32", "207d3364b93f45e5a1745da484f1407b", "f401fcdf0fef44939dea006049389457", "49140feaace54bdbad6af698d3b3f7b1", "dfc3bce7fea5444683961aca37e37711", "da0642a6b3c24f98b60bf6f984d680a6", "62ad4b9a8b8e4d24a1c0140b8ce9a857", "fa421a4b5a5c426998aa106e99558ee6", "5728a3a88f3d45028adb0c9b86e0020b"]} id="Vk4mmOpSwTI1" executionInfo={"status": "ok", "timestamp": 1636042903381, "user_tz": -330, "elapsed": 4452, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1fa07f55-29a7-4d12-8411-1439f827f7f6"
counter = defaultdict(lambda: 0)
for i in tqdm(range(df_new.shape[0])):
    user_exposed_history = df_new.at[i, 'exposed_items'].split(',')
    labels = df_new.at[i, 'labels'].split(',')
    for idx, itemid in enumerate(user_exposed_history):
        if labels[idx] == '1':
            counter[itemid] += 1
counter = dict(counter)
counter_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="i1ihPGCtwVcB" executionInfo={"status": "ok", "timestamp": 1636042905747, "user_tz": -330, "elapsed": 2375, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="12b368ae-b01b-4dac-ff61-e2dfb11a4057"
counter_sorted_session1 = [s for s in counter_sorted if 1 <= int(s[0]) <= 39]
counter_sorted_session2 = [s for s in counter_sorted if 40 <= int(s[0]) <= 147]
counter_sorted_session3 = [s for s in counter_sorted if 148 <= int(s[0]) <= 381]

nums_to_plot = counter_sorted_session1
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item bought? (Session 1)')
plt.xlabel('itemid')
plt.ylabel('item bought times')
plt.tight_layout()
plt.show()

nums_to_plot = counter_sorted_session2[:40]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item bought? (Session 2)')
plt.xlabel('itemid')
plt.ylabel('item bought times')
plt.tight_layout()
plt.show()

nums_to_plot = counter_sorted_session3[:40]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1] for n in nums_to_plot]
plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
plt.title('How many times is item bought? (Session 3)')
plt.xlabel('itemid')
plt.ylabel('item bought times')
plt.tight_layout()
plt.show()
```

<!-- #region id="cLPgSXrNwYgZ" -->
### Expose vs Bought
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["040b288af4884fc38493eaa5b67c5dcf", "ad3cbcd79eac4ee0ba70a1c2a9dbb63d", "21a54afa7092415b80b4aa144670834f", "f46dc1e9f9764c40a3c265c7fe3113da", "87a54f3c2fce4c0789074459d76e4e76", "4253145e867e4c528f52d45de88baf45", "8fe4592c8e974363884399d97b3a362f", "237fe46de5284ed5862a6969702a31ea", "e6a000e3476c43b7aedc8af0d29edebd", "955380c3090e4224b532738221cc658c", "4539dd97190f4f548199c628062f237c"]} id="3vtu561wwdoM" executionInfo={"status": "ok", "timestamp": 1636042946815, "user_tz": -330, "elapsed": 7707, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bd3f37b1-ce53-4e3c-aba4-dc7b47d0534a"
## expose should only be counted when users are in that session
# s1: 0~2, s2: 3~5, s3: 6~9

counter = defaultdict(lambda: [0, 0])
for i in tqdm(range(df_new.shape[0])):
    user_exposed_history = df_new.at[i, 'exposed_items']
    num_bought = sum([int(l) for l in df_new.at[i, 'labels'].split(',')])
    labels = df_new.at[i, 'labels'].split(',')
    for idx, itemid in enumerate(user_exposed_history.split(',')):
        if 0 <= idx <= 2:
            counter[itemid][0] += 1
            if labels[idx] == '1':
                counter[itemid][1] += 1
        elif 3 <= idx <= 5 and num_bought >= 3: 
            counter[itemid][0] += 1
            if labels[idx] == '1':
                counter[itemid][1] += 1
        elif 6 <= idx <= 8 and num_bought >= 6:
            counter[itemid][0] += 1
            if labels[idx] == '1':
                counter[itemid][1] += 1
counter = dict(counter)
counter_sorted = sorted(counter.items(), key=lambda x: x[1][0], reverse=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="GcLxvRAswfSQ" executionInfo={"status": "ok", "timestamp": 1636042949996, "user_tz": -330, "elapsed": 3205, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8b05400b-72d9-4b68-b9f4-1c2bf9a2cb58"
counter_sorted_session1 = [s for s in counter_sorted if 1 <= int(s[0]) <= 39]
counter_sorted_session2 = [s for s in counter_sorted if 40 <= int(s[0]) <= 147]
counter_sorted_session3 = [s for s in counter_sorted if 148 <= int(s[0]) <= 381]

nums_to_plot = counter_sorted_session1[:40]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1][0] for n in nums_to_plot]
bought_cnt = [n[1][1] for n in nums_to_plot]
X = np.arange(len(itemids))
fig = plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, click_cnt, color = 'b', width = 0.25)
ax.bar(X + 0.20, bought_cnt, color = 'r', width = 0.25)
plt.xticks(X, itemids)
plt.title('How many times is item exposed, bought? (Session1)')
plt.xlabel('itemid')
plt.ylabel('item exposed, bought times')
plt.show()

nums_to_plot = counter_sorted_session2[:40]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1][0] for n in nums_to_plot]
bought_cnt = [n[1][1] for n in nums_to_plot]
X = np.arange(len(itemids))
fig = plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, click_cnt, color = 'b', width = 0.25)
ax.bar(X + 0.20, bought_cnt, color = 'r', width = 0.25)
plt.xticks(X, itemids)
plt.title('How many times is item exposed, bought? (Session2)')
plt.xlabel('itemid')
plt.ylabel('item exposed, bought times')
plt.show()

nums_to_plot = counter_sorted_session3[:40]
itemids = [n[0] for n in nums_to_plot]
click_cnt = [n[1][0] for n in nums_to_plot]
bought_cnt = [n[1][1] for n in nums_to_plot]
X = np.arange(len(itemids))
fig = plt.figure(figsize=(15, 5))
plt.bar(itemids, click_cnt, width=0.2)
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, click_cnt, color = 'b', width = 0.25)
ax.bar(X + 0.20, bought_cnt, color = 'r', width = 0.25)
plt.xticks(X, itemids)
plt.title('How many times is item exposed, bought? (Session3)')
plt.xlabel('itemid')
plt.ylabel('item exposed, bought times')
plt.show()
```

<!-- #region id="sTMxsHLzwqrI" -->
### Expose vs Bought by Session
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["7fda48dee4f24568a8b63746a0d4e972", "56c5cc76044441589f9527da40c3beb7", "6591888d341d45a28d48b5afdd46d3b5", "3303604a0ae140eab3c7e051cfc345f3", "ec017f34d7104ffb93cb43a0f8e0f737", "76d08a0ca1ed4ce381c3a3cca1326041", "777a02ef205040418da00eec911bc244", "d3ef7e91eef14605b5d5555b66379eac", "89bca81ed1d74f139b5411ff7c999c2f", "034a5e16e2a541dcb02e988a3b475399", "c22887ea972c44cfab929080ed2110b5"]} id="75tYoc5uwsXP" executionInfo={"status": "ok", "timestamp": 1636043009198, "user_tz": -330, "elapsed": 8262, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="941e91f6-248d-4d89-80e0-16ae403c88dc"
## expose should only be counted when users are in that session
# s1: 0~2, s2: 3~5, s3: 6~9

counter = defaultdict(lambda: [0, 0, 0])
for i in tqdm(range(df_new.shape[0])):
    user_exposed_history = df_new.at[i, 'exposed_items']
    num_bought = sum([int(l) for l in df_new.at[i, 'labels'].split(',')])
    labels = df_new.at[i, 'labels'].split(',')
    for idx, itemid in enumerate(user_exposed_history.split(',')):
        if 0 <= idx <= 2:
            counter[itemid][0] += 1
            if labels[idx] == '1':
                counter[itemid][1] += 1
        elif 3 <= idx <= 5 and num_bought >= 3: 
            counter[itemid][0] += 1
            if labels[idx] == '1':
                counter[itemid][1] += 1
        elif 6 <= idx <= 8 and num_bought >= 6:
            counter[itemid][0] += 1
            if labels[idx] == '1':
                counter[itemid][1] += 1
counter = dict(counter)
for k, v in counter.items():
    counter[k][2] = counter[k][1] / counter[k][0]
counter_sorted = sorted(counter.items(), key=lambda x: x[1][2], reverse=True)

counter_sorted_session1 = [s for s in counter_sorted if 1 <= int(s[0]) <= 39]
counter_sorted_session2 = [s for s in counter_sorted if 40 <= int(s[0]) <= 147]
counter_sorted_session3 = [s for s in counter_sorted if 148 <= int(s[0]) <= 381]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="XuCokMCXwwbJ" executionInfo={"status": "ok", "timestamp": 1636043052725, "user_tz": -330, "elapsed": 6404, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="598192a0-f501-4292-8348-9dd8b4a75a74"
nums_to_plot = counter_sorted_session1
itemids = [n[0] for n in nums_to_plot]
expose_cnt = [n[1][0] for n in nums_to_plot]
bought_cnt = [n[1][1] for n in nums_to_plot]
bought_expose_ratio = [n[1][2] for n in nums_to_plot]
X = np.arange(len(itemids))
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, expose_cnt, color = 'b', width = 0.2)
ax.bar(X + 0.20, bought_cnt, color = 'r', width = 0.2)
plt.xticks(X, itemids)
plt.title('How many times is item exposed, bought? (Session1) sorted by ratio')
plt.xlabel('itemid')
plt.ylabel('item exposed, bought times')
ax2 = ax.twinx()
ax2.bar(X + 0.40, bought_expose_ratio, color = 'g', width = 0.2)
ax2.set_ylim([0, 1])
ax2.set_ylabel('ratio: bought/exposed', color='b')
# plt.yscale('log')
plt.show()


nums_to_plot = counter_sorted_session2
itemids = [n[0] for n in nums_to_plot]
expose_cnt = [n[1][0] for n in nums_to_plot]
bought_cnt = [n[1][1] for n in nums_to_plot]
bought_expose_ratio = [n[1][2] for n in nums_to_plot]
X = np.arange(len(itemids))
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, expose_cnt, color = 'b', width = 0.2)
ax.bar(X + 0.20, bought_cnt, color = 'r', width = 0.2)
plt.xticks(X, itemids)
plt.title('How many times is item exposed, bought? (Session2) sorted by ratio')
plt.xlabel('itemid')
plt.ylabel('item exposed, bought times')
ax2 = ax.twinx()
ax2.bar(X + 0.40, bought_expose_ratio, color = 'g', width = 0.2)
ax2.set_ylim([0, 1])
ax2.set_ylabel('ratio: bought/exposed', color='b')
# plt.yscale('log')
plt.show()


nums_to_plot = counter_sorted_session3
itemids = [n[0] for n in nums_to_plot]
expose_cnt = [n[1][0] for n in nums_to_plot]
bought_cnt = [n[1][1] for n in nums_to_plot]
bought_expose_ratio = [n[1][2] for n in nums_to_plot]
X = np.arange(len(itemids))
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, expose_cnt, color = 'b', width = 0.2)
ax.bar(X + 0.20, bought_cnt, color = 'r', width = 0.2)
plt.xticks(X, itemids)
plt.title('How many times is item exposed, bought? (Session3) sorted by ratio')
plt.xlabel('itemid')
plt.ylabel('item exposed, bought times')
# plt.yscale('log')
ax2 = ax.twinx()
ax2.bar(X + 0.40, bought_expose_ratio, color = 'g', width = 0.2)
ax2.set_ylim([0, 1])
ax2.set_ylabel('ratio: bought/exposed', color='b')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="AoQtTN_OwyS8" executionInfo={"status": "ok", "timestamp": 1636043140160, "user_tz": -330, "elapsed": 3594, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bd206f0f-620d-40df-81b3-343ce4cb2636"
nums_to_plot = counter_sorted_session1[:40]
itemids = [n[0] for n in nums_to_plot]
expose_cnt = [n[1][0] for n in nums_to_plot]
bought_cnt = [n[1][1] for n in nums_to_plot]
bought_expose_ratio = [n[1][2] for n in nums_to_plot]
X = np.arange(len(itemids))
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, expose_cnt, color = 'b', width = 0.2)
ax.bar(X + 0.20, bought_cnt, color = 'r', width = 0.2)
plt.xticks(X, itemids)
plt.title('How many times is item exposed, bought? (Session1) sorted by ratio')
plt.xlabel('itemid')
plt.ylabel('item exposed, bought times')
ax2 = ax.twinx()
ax2.bar(X + 0.40, bought_expose_ratio, color = 'g', width = 0.2)
ax2.set_ylim([0, 1])
ax2.set_ylabel('ratio: bought/exposed', color='b')
# plt.yscale('log')
plt.show()


nums_to_plot = counter_sorted_session2[:40]
itemids = [n[0] for n in nums_to_plot]
expose_cnt = [n[1][0] for n in nums_to_plot]
bought_cnt = [n[1][1] for n in nums_to_plot]
bought_expose_ratio = [n[1][2] for n in nums_to_plot]
X = np.arange(len(itemids))
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, expose_cnt, color = 'b', width = 0.2)
ax.bar(X + 0.20, bought_cnt, color = 'r', width = 0.2)
plt.xticks(X, itemids)
plt.title('How many times is item exposed, bought? (Session2) sorted by ratio')
plt.xlabel('itemid')
plt.ylabel('item exposed, bought times')
ax2 = ax.twinx()
ax2.bar(X + 0.40, bought_expose_ratio, color = 'g', width = 0.2)
ax2.set_ylim([0, 1])
ax2.set_ylabel('ratio: bought/exposed', color='b')
# plt.yscale('log')
plt.show()


nums_to_plot = counter_sorted_session3[:40]
itemids = [n[0] for n in nums_to_plot]
expose_cnt = [n[1][0] for n in nums_to_plot]
bought_cnt = [n[1][1] for n in nums_to_plot]
bought_expose_ratio = [n[1][2] for n in nums_to_plot]
X = np.arange(len(itemids))
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, expose_cnt, color = 'b', width = 0.2)
ax.bar(X + 0.20, bought_cnt, color = 'r', width = 0.2)
plt.xticks(X, itemids)
plt.title('How many times is item exposed, bought? (Session3) sorted by ratio')
plt.xlabel('itemid')
plt.ylabel('item exposed, bought times')
ax2 = ax.twinx()
ax2.bar(X + 0.40, bought_expose_ratio, color = 'g', width = 0.2)
ax2.set_ylim([0, 1])
ax2.set_ylabel('ratio: bought/exposed', color='b')
# plt.yscale('log')
plt.show()
```

<!-- #region id="16KOOpGAxMOV" -->
### Label Analysis
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 327, "referenced_widgets": ["1d1998757d5a42d0bc124e2ad224fe33", "234f2589d3594f14ab4b32cc989b6a94", "55e03a8ea4894b06ae51f43bdd9a1593", "1416476dbcf042769a8b56e1fb39ac7b", "ac25358c9542487e821fb3d9bb002369", "eebcdd23f792428892dc84fb72ce400b", "a42cee3f918c413cafd3c42c27020a23", "a27201213cb74a9091d8d3aa7fb77529", "f76a7248e9b14e758fb61d110000ab25", "2b555261632d46599ce91d130989dc76", "1035f2fbfa704e2ea82d56e228cba160"]} id="xKitqhUA0Unh" executionInfo={"status": "ok", "timestamp": 1636044002688, "user_tz": -330, "elapsed": 6270, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d49600cb-d0c3-44ce-d732-30ba0f2c09a1"
for i in tqdm(range(df_new.shape[0])):
    labels = df_new.at[i, 'labels'].split(',')
    df_new.at[i, 'cnt_labels'] = sum([int(l) for l in labels])

arr = np.array(df_new['cnt_labels'])
plt.hist(arr, color='blue', edgecolor='black', bins=100)
plt.title('Histogram of # of items users bought')
plt.xticks(range(10), range(10))
plt.xlabel('# of items users bought')
plt.ylabel('Count')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ip40BO9f0gBD" executionInfo={"status": "ok", "timestamp": 1636044004828, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e91b7bb7-b5bd-486a-ef46-e5383a8c1449"
arr = np.array(df_new['cnt_labels'])
for i in range(10):
    print('There are', len(arr[arr == i]), 'users have bout', i, 'items')
```
