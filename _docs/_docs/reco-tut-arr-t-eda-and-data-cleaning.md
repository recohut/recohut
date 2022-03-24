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

```python id="-UOOzCs9ukul" executionInfo={"status": "ok", "timestamp": 1628700920220, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-arr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="_leacZxaIc82" executionInfo={"status": "ok", "timestamp": 1628700924130, "user_tz": -330, "elapsed": 3929, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="902f3c07-0b45-4b41-92df-fc33660fea65"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="LLzPE36xIc8-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628706211336, "user_tz": -330, "elapsed": 1078, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6ab58ca3-1f2a-434d-dc89-0e2c1180a044"
!git status
```

```python id="zL4F4MHaIc8_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628706215672, "user_tz": -330, "elapsed": 1867, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fa8e2283-7fd4-4607-8999-9af8ce7f12a9"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="gYJpj93uIo1B" -->
---
<!-- #endregion -->

```python id="gCgXr2x3Jd5s" executionInfo={"status": "ok", "timestamp": 1628704887574, "user_tz": -330, "elapsed": 1154, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
from tqdm import tqdm
import pickle
from numpy import log, sqrt, log2, ceil, exp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import warnings
warnings.filterwarnings('ignore')
```

```python id="kQtCMxl1JDrY" executionInfo={"status": "ok", "timestamp": 1628700979186, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_sequences(df:pd.DataFrame, target:str, group_by:list, sort_by:str=None, sort:bool=False, min_seq_len=1) -> pd.Series:
    """Groups a DataFrame by features and aggregates target feature into a Series of lists."""
    clone = df.copy()
    if sort:
        clone.sort_values(by=sort_by, inplace=True)
    group = clone.groupby(by=group_by)
    sequences = group[target].apply(list)
    sequences = sequences[sequences.apply(lambda x: len(x)) >= min_seq_len]    # Filter out length 0 sequences
    return sequences
```

```python id="U6nJWEPkJG4x" executionInfo={"status": "ok", "timestamp": 1628700980110, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def integer_encoding(df:pd.DataFrame, cols:list, min_int=0, drop_old=False, monotone_mapping:bool=False):
    """Returns updated DataFrame and inverse mapping dictionary."""
    clone = df.copy()
    id_maps = dict()
    inv_maps = dict()
    for col in cols:
        # If list-valued
        if type(clone.iloc[0][col]) == list:
            # Get unique values and sort
            unique_values = clone[col].explode().unique()
            num_unique = unique_values.size
            if monotone_mapping:
                unique_values.sort()
            # Generate dictionary maps
            id_map = dict()
            inv_map = dict()
            for i in range(num_unique):
                id_map[unique_values[i]] = i + min_int
                inv_map[i + min_int] = unique_values[i]
            id_maps[col] = id_map
            inv_maps = inv_map
            # Encoding
            if drop_old:
                clone[col] = clone[col].apply(lambda x: [id_map[i] for i in x])
            else:
                col_reidx = col + "_reidx"
                clone[col_reidx] = clone[col].apply(lambda x: [id_map[i] for i in x])
        else:
            # Get unique values and sort
            unique_values = clone[col].unique()
            num_unique = unique_values.size
            if monotone_mapping:
                unique_values.sort()
            # Generate dictionary maps
            id_map = dict()
            inv_map = dict()
            for i in range(num_unique):
                id_map[unique_values[i]] = i + min_int
                inv_map[i + min_int] = unique_values[i]
            id_maps[col] = id_map
            inv_maps[col] = inv_map
            # Encoding
            if drop_old:
                clone[col] = clone[col].map(id_map)
            else:
                col_reidx = col + "_reidx"
                clone[col_reidx] = clone[col].map(id_map)
    return clone, id_maps, inv_maps
```

```python id="CTmWBIChJSMm" executionInfo={"status": "ok", "timestamp": 1628701013885, "user_tz": -330, "elapsed": 411, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def multiclass_list_encoding(df:pd.DataFrame, cols:list, drop_old=False):
    clone = df.copy()
    # For index conjugation to make querying easy
    index_map = dict()
    inv_map = dict()
    for i, idx in enumerate(clone.index):
        index_map[idx] = i
        inv_map[i] = idx
    clone.index = clone.index.map(index_map)
    for col in cols:
        # If list-valued
        if type(clone.iloc[0][col]) == list:
            categories = clone[col].explode().unique().tolist()
            categories.sort()
            # Init one-hot columns
            for cat in categories:
                cat_col = col + "_is_" + str(cat)
                clone[cat_col] = 0
            # Define encoding function to be vectorized
            def f(row):
                row_cats = row[col]     # type(row_cats) == list
                for row_cat in row_cats:
                    row_cat_col = col + "_is_" + str(row_cat)
                    idx = row.name
                    clone.loc[idx, row_cat_col] = 1
            clone.apply(f, axis=1)
        # If not list-valued
        else:
            categories = clone[col].unique().tolist()
            categories.sort()
            # Init one-hot columns
            for cat in categories:
                cat_col = col + "_is_" + str(cat)
                clone[cat_col] = 0
            # Define encoding function to be vectorized
            def g(row):
                row_cat = row[col]
                row_cat_col = col + "_is_" + str(row_cat)
                idx = row.name
                clone.loc[idx, row_cat_col] = 1
            clone.apply(g, axis=1)
    if drop_old:
        clone.drop(labels=cols, axis=1, inplace=True)
    clone.index = clone.index.map(inv_map)
    return clone
```

```python id="ysVsyJ6gJWCT" executionInfo={"status": "ok", "timestamp": 1628701029370, "user_tz": -330, "elapsed": 1333, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def pool_encodings_from_sequences(sequences:pd.Series, pool_from: pd.DataFrame):
    """Inputs a Pandas Series `sequences` valued in lists of indices from `pool_from`.
    Outputs a Pandas DataFrame with columns from `pool_from` and indices from `sequences`
    with values given as a mean over `pool_from` rows supplied from `sequences`."""
    encoded = pd.DataFrame(index=sequences.index, columns=pool_from.columns, dtype='float64')
    seq_df = sequences.to_frame()
    col = seq_df.columns[0]
    def f(row):
        seq = row[col]
        encoded.loc[row.name] = pool_from[pool_from.index.isin(seq)].mean(axis=0)
        return None
    seq_df.apply(f, axis=1)
    return encoded
```

```python id="xuAR4IfTJXh5" executionInfo={"status": "ok", "timestamp": 1628701033470, "user_tz": -330, "elapsed": 525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_inputs_from_sequences(sequences:pd.Series, customers:pd.DataFrame, vendors:pd.DataFrame):
    out = torch.zeros((1, 2 * len(customers.columns)))  # For shape
    seq_df = sequences.to_frame()
    col = seq_df.columns[0]
    def f(row):
        seq = row[col]
        c_tensor = torch.tensor(customers.loc[row.name])
        for vendor in seq:
            v_tensor = torch.tensor(vendors.iloc[vendor])
            pair = torch.cat((c_tensor, v_tensor)).view(1, -1)
            out = torch.cat((out, pair), axis=0)
        return None
    seq_df.apply(f, axis=1)
    return out[1:]
```

```python id="rUmRPX7vJaUu" executionInfo={"status": "ok", "timestamp": 1628701051765, "user_tz": -330, "elapsed": 686, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def pandas_sequences_to_tensor(sequences:pd.Series, window:int=5):
    """Convert pandas series of sequences to pytorch tensor of padded sequences"""
    def left_pad_list(L):
        nonlocal window
        num_zeros = window - len(L)
        return ([0] * num_zeros) + L
    def get_windows(L):
        nonlocal window
        out = list()
        for i in range(1, len(L)+1):
            if i <= window:
                out.append(left_pad_list(L[:i]))
            else:
                out.append(L[i-window:i])
        return out
    padded_sequences = torch.stack(sequences.apply(get_windows).explode().apply(torch.tensor).tolist(), axis=0)
    return padded_sequences
```

```python id="p2Q3DRG_IpJB" executionInfo={"status": "ok", "timestamp": 1628701057924, "user_tz": -330, "elapsed": 646, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.customer = sequences[:, :-1]
        self.vendor = sequences[:, -1:].view(-1)
    def __len__(self):
        return len(self.vendor)
    def __getitem__(self, idx):
        return self.customer[idx,:], self.vendor[idx]
```

```python id="AclZ46HeJu5A" executionInfo={"status": "ok", "timestamp": 1628701190246, "user_tz": -330, "elapsed": 1242, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
vendors = pd.read_parquet('./data/bronze/vendors.parquet.gz')
orders = pd.read_parquet('./data/bronze/orders.parquet.gz')
train_customers = pd.read_parquet('./data/bronze/train_customers.parquet.gz')
train_locations = pd.read_parquet('./data/bronze/train_locations.parquet.gz')
test_customers = pd.read_parquet('./data/bronze/test_customers.parquet.gz')
test_locations = pd.read_parquet('./data/bronze/test_locations.parquet.gz')
```

<!-- #region id="9EvGJ9PyJ-AC" -->
## Orders
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 343} id="tkxg61YGKdj5" executionInfo={"status": "ok", "timestamp": 1628701325606, "user_tz": -330, "elapsed": 538, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="62f3a320-db3e-4c99-9e85-91cd1964a51c"
orders.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6NTz1_7GKfMu" executionInfo={"status": "ok", "timestamp": 1628701375086, "user_tz": -330, "elapsed": 843, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="801336eb-7720-431a-fe00-63d22e32ea09"
orders.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 716} id="b-4Yjo-oKq9q" executionInfo={"status": "ok", "timestamp": 1628701631635, "user_tz": -330, "elapsed": 4029, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b3a02ef7-5528-4d71-e621-d70944eb5656"
fig, ax = plt.subplots(figsize=(15,12))
orders.hist(ax=ax)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 317} id="vvB11T3xK5ef" executionInfo={"status": "ok", "timestamp": 1628702530702, "user_tz": -330, "elapsed": 712, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="08b390cf-3bd4-4a1d-9587-449538db8ffa"
orders.describe().round(1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 244} id="MU5oqngvOpcr" executionInfo={"status": "ok", "timestamp": 1628702551737, "user_tz": -330, "elapsed": 1658, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c8ab7f27-2cd7-4013-f4aa-e35eb91d8e44"
orders.describe(include='O')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 327} id="HA4Io-J-QGG6" executionInfo={"status": "ok", "timestamp": 1628703237014, "user_tz": -330, "elapsed": 2174, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="734b4ab7-5849-4c39-dda9-1d2f38c0d2e7"
fig, ax = plt.subplots(1,4, figsize=(20,5))
train_orders['is_favorite'].value_counts().plot(kind='bar', ax=ax[0], title='is_favorite');
train_orders['is_rated'].value_counts().plot(kind='bar', ax=ax[1], title='is_rated');
train_orders['LOCATION_TYPE'].value_counts().plot(kind='bar', ax=ax[2], title='location_type');
train_orders['promo_code'].value_counts()[:20].plot(kind='bar', ax=ax[3], rot=75, title='promo_code');
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="in_to5l_ZziO" executionInfo={"status": "ok", "timestamp": 1628705350068, "user_tz": -330, "elapsed": 445, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="074e503c-6bcb-49f4-80a2-673f126621b5"
test_customers.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="-I5iKTMeah_M" executionInfo={"status": "ok", "timestamp": 1628705534194, "user_tz": -330, "elapsed": 475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5dbe5595-15f0-4f29-8f08-926b9e21c13e"
test_customers.dtypes
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="o5mFYu8tabtU" executionInfo={"status": "ok", "timestamp": 1628705512968, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9b047134-1550-48cd-f08d-97330f5a4619"
train_customers.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="iF59b54Lae2_" executionInfo={"status": "ok", "timestamp": 1628705595757, "user_tz": -330, "elapsed": 419, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d1e54015-8ad0-4c78-a471-4ce8faa04997"
train_customers.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="UlTEdUFiZ3Eg" executionInfo={"status": "ok", "timestamp": 1628705604343, "user_tz": -330, "elapsed": 458, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="19a79ace-16a8-4256-e106-9c9a4f8f375d"
orders.shape
```

```python id="1dsvvSedbEro" executionInfo={"status": "ok", "timestamp": 1628705704225, "user_tz": -330, "elapsed": 460, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
_x = test_customers['akeed_customer_id'].tolist()
_y = orders['customer_id'].tolist()
```

```python colab={"base_uri": "https://localhost:8080/"} id="dknErA6yZrW1" executionInfo={"status": "ok", "timestamp": 1628705720161, "user_tz": -330, "elapsed": 437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e83cda20-e050-44ed-ca9f-ff4d81e87898"
len(_x), len(_y)
```

```python colab={"base_uri": "https://localhost:8080/"} id="NblCMKq3bdxP" executionInfo={"status": "ok", "timestamp": 1628705783001, "user_tz": -330, "elapsed": 385, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3fdb27f3-2833-49e6-8ccf-a1196f3852a0"
list(set(_x).intersection(_y))
```

```python colab={"base_uri": "https://localhost:8080/"} id="Vt1lMUxTPKUb" executionInfo={"status": "ok", "timestamp": 1628702667264, "user_tz": -330, "elapsed": 431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7cab02cd-e668-41e3-8881-d511f81eef8d"
# train and test orders
train_orders = orders[orders['customer_id'].isin(train_customers['akeed_customer_id'])]
test_orders = orders[orders['customer_id'].isin(test_customers['akeed_customer_id'])]

# remove duplicate customers and their orders
x = train_customers.groupby('akeed_customer_id').size()
duplicate_train_customers = train_customers[train_customers['akeed_customer_id'].isin(x[x>1].index)]['akeed_customer_id'].unique()
train_customers = train_customers[~train_customers['akeed_customer_id'].isin(duplicate_train_customers)]
train_orders = train_orders[~train_orders['customer_id'].isin(duplicate_train_customers)]

# number of train and test orders
num_train_orders = orders[orders['customer_id'].isin(train_customers['akeed_customer_id'])].shape[0]
num_test_orders = orders[orders['customer_id'].isin(test_customers['akeed_customer_id'])].shape[0]
print(f'Num Orders: {orders.shape[0]}\nNum Train: {num_train_orders}\nNum Test: {num_test_orders}')
```

<!-- #region id="jv-XKUb8PmwS" -->
## Vendors
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="LQAfADQ1SEwp" executionInfo={"status": "ok", "timestamp": 1628703331549, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ca829145-eab5-4b26-92c7-ebb9dbdf80e6"
vendors.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="b1P_26VdSJCN" executionInfo={"status": "ok", "timestamp": 1628703337056, "user_tz": -330, "elapsed": 491, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7fc26489-3eb7-4156-f5a7-dbe54009c2ce"
vendors.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 648} id="rXJsJYn8SKWO" executionInfo={"status": "ok", "timestamp": 1628703428030, "user_tz": -330, "elapsed": 4700, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cd1c3824-099a-4cd3-e7ea-b9ce975da604"
fig, ax = plt.subplots(figsize=(18,12))
vendors.hist(ax=ax)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 317} id="w__hvknFSbAn" executionInfo={"status": "ok", "timestamp": 1628703498453, "user_tz": -330, "elapsed": 618, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="28527676-a763-48af-8ba3-eef5daea70f1"
vendors.describe().round(1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 210} id="mj6Pp_mWSxty" executionInfo={"status": "ok", "timestamp": 1628703518049, "user_tz": -330, "elapsed": 540, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6581b1ec-6b54-40df-f5e5-ea6f4d41e670"
vendors.describe(include='O')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} id="zHz9luksS2gE" executionInfo={"status": "ok", "timestamp": 1628703900988, "user_tz": -330, "elapsed": 659, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3874194a-acb0-4585-8707-6c6c010071be"
# reindex vendor id
vendors.sort_values(by='id')
vendors, v_id_maps, v_inv_maps = integer_encoding(df=vendors, cols=['id'], min_int=1, drop_old=True, monotone_mapping=True)
v_id_map, v_inv_map = v_id_maps['id'], v_inv_maps['id']

# set id column to index
vendors.set_index('id', inplace=True)

# cleaning vendor categories
vendors[(vendors['vendor_category_en'] == "Sweets & Bakes") & (vendors['vendor_category_id'] == 3.0)].shape[0]
vendors[(vendors['vendor_category_en'] == "Sweets & Bakes") & (vendors['vendor_category_id'] == 2.0)]

# fix incorrect vendor_category_id
vendors.loc[28, 'vendor_category_id'] = 3.0

# cleaning vendor tags
# fill na with -1 and strip unnecessary characters
vendors['primary_tags'] = vendors['primary_tags'].fillna("{\"primary_tags\":\"-1\"}").apply(lambda x: int(str(x).split("\"")[3]))
# fill na with -1 and turn vendor_tag into list-valued
vendors['vendor_tag'] = vendors['vendor_tag'].fillna(str(-1)).apply(lambda x: x.split(",")).apply(lambda x: [int(i) for i in x])

# get unique vendor tags and map values to range(len(vendor_tags))
vendor_tags = [int(i) for i in vendors['vendor_tag'].explode().unique()]
vendor_tags.sort()
vendor_map = dict()
for i, tag in enumerate(vendor_tags): vendor_map[tag] = i
vendors['vendor_tag'] = vendors['vendor_tag'].apply(lambda tags: [vendor_map[tag] for tag in tags])

# combine status and verified features
vendors['status_and_verified'] = vendors['status'] * vendors['verified']

# print some rows
vendors.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} id="8YoRDufTUT3-" executionInfo={"status": "ok", "timestamp": 1628704051838, "user_tz": -330, "elapsed": 472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ae682188-cfac-444f-d281-57b72c635256"
# create some order-based features
# add num_orders, amt_sales, and avg_sale as new columns in vendor table
train_orders_vendor_grp = train_orders.groupby(by=['vendor_id'])
train_orders_per_vendor = train_orders_vendor_grp['akeed_order_id'].count().rename('num_orders')
train_orders_per_vendor.index = train_orders_per_vendor.index.map(v_id_map)
train_grand_total_per_vendor = train_orders_vendor_grp['grand_total'].sum().rename('amt_sales')
train_grand_total_per_vendor.index = train_grand_total_per_vendor.index.map(v_id_map)

# test_orders_vendor_grp = test_orders.groupby(by=['vendor_id'])
# test_orders_per_vendor = test_orders_vendor_grp['akeed_order_id'].count().rename('num_orders')
# test_orders_per_vendor.index = test_orders_per_vendor.index.map(v_id_map)
# test_grand_total_per_vendor = test_orders_vendor_grp['grand_total'].sum().rename('amt_sales')
# test_grand_total_per_vendor.index = test_grand_total_per_vendor.index.map(v_id_map)

vendors = vendors.merge(train_orders_per_vendor, how='left', left_on='id', right_index=True)
vendors = vendors.merge(train_grand_total_per_vendor, how='left', left_on='id', right_index=True)
vendors['avg_sale'] = vendors['amt_sales'] / vendors['num_orders']

# save most popular vendors
popular_vendors = vendors['num_orders'].sort_values(ascending=False)
with open("./data/silver/popular_vendors.pkl", "wb") as file:
    pickle.dump(popular_vendors, file)

vendors['num_orders_log3'] = vendors['num_orders'].apply(log).apply(log).apply(log)
vendors['amt_sales_log3'] = vendors['amt_sales'].apply(log).apply(log).apply(log)
vendors['avg_sale_log'] = vendors['avg_sale'].apply(log)

vendors.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="TjxI2ajOU40d" executionInfo={"status": "ok", "timestamp": 1628704226370, "user_tz": -330, "elapsed": 428, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d0540617-c470-4517-dd30-71611f9994d1"
# transforming location outliers
orders_55 = train_orders[train_orders['vendor_id'] == 55]
orders_99 = train_orders[train_orders['vendor_id'] == 99]

orders_55 = orders_55.merge(train_locations, how='left', left_on=['customer_id', 'LOCATION_NUMBER'], right_on=['customer_id', 'location_number'])
orders_99 = orders_99.merge(train_locations, how='left', left_on=['customer_id', 'LOCATION_NUMBER'], right_on=['customer_id', 'location_number'])
lat55 = vendors[vendors.index == 55].latitude.item()
long55 = vendors[vendors.index == 55].longitude.item()
lat99 = vendors[vendors.index == 99].latitude.item()
long99 = vendors[vendors.index == 99].longitude.item()

print(f'55 actual: \tLat = {lat55:.3f}, Long = {long55:.3f}')
print(f'55 estimate: \tLat = {orders_55.latitude.median():.3f}, Long = {orders_55.longitude.median():.3f}')
print(f'99 actual: \tLat = {lat99:.3f}, Long = {long99:.3f}')
print(f'99 estimate: \tLat = {orders_99.latitude.median():.3f}, Long = {orders_99.longitude.median():.3f}')

# aggregate # orders, $ sales, and avg spent by customer location
# (customers can have multiple locations registered to themselves)
orders_location_grp = train_orders.groupby(['customer_id', 'LOCATION_NUMBER'])
orders_per_location = orders_location_grp['akeed_order_id'].count().rename('num_orders')    # multi index: [customer_id, LOCATION_NUMBER]
sales_per_location = orders_location_grp['grand_total'].sum().rename('amt_spent')           # multi index: [customer_id, LOCATION_NUMBER]

train_locations = train_locations.merge(sales_per_location, how='left', left_on=['customer_id', 'location_number'], right_index=True)
train_locations = train_locations.merge(orders_per_location, how='left', left_on=['customer_id', 'location_number'], right_index=True)
train_locations['avg_spend'] = train_locations['amt_spent'] / train_locations['num_orders']

# filter locations which have not been ordered from
train_locations = train_locations[train_locations['num_orders'] != 0]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 350} id="yfTK5LqFVjax" executionInfo={"status": "ok", "timestamp": 1628704275200, "user_tz": -330, "elapsed": 647, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d971faea-6df6-4f4e-a7de-140774d3ab2b"
# map out customer locations and vendor locations
plt.figure(figsize=(6, 5))
plt.scatter(x=train_locations.longitude, y=train_locations.latitude, label='Customers', marker='s', alpha=0.2)
plt.scatter(x=vendors.longitude, y=vendors.latitude, label='Vendors', marker='*', alpha=0.5, s=vendors['num_orders']/5, c=vendors['avg_sale'], cmap='plasma')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='lower right')
plt.colorbar(label='$ Avg Sale')
plt.title('Customer + Vendor Locations')
plt.show()
```

<!-- #region id="bs0UdgDmV5uM" -->
Outliers in location are probably a mistake (GPS error?)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 350} id="yGPq_4zeVogz" executionInfo={"status": "ok", "timestamp": 1628704350642, "user_tz": -330, "elapsed": 949, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="471ca489-b9e2-4138-fb86-3c0713dddfb1"
# Zoom in on area close with most activity
# Marker Size = # Orders and Color = $ Grand Total

lo = -5
hi = 5

filt1 = (lo < train_locations['longitude']) & (train_locations['longitude'] < hi)
filt2 = (lo < vendors['longitude']) & (vendors['longitude'] < hi)
train_locations_cut = train_locations[filt1]
vendors_cut = vendors[filt2]

plt.figure(figsize=(6, 5))
plt.scatter(x=train_locations_cut.longitude, y=train_locations_cut.latitude, label='Customers', marker='s', alpha=0.1)
plt.scatter(x=vendors_cut.longitude, y=vendors_cut.latitude, label='Vendors', marker='*', alpha=0.5, s=vendors_cut['num_orders']/7, c=vendors_cut['avg_sale'], cmap='plasma')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='lower right')
plt.colorbar(label='$ Avg Sale')
plt.title('Customer + Vendor Locations (Zoomed)')
plt.show()
```

```python id="YYioGzOSWBrC"
# Define outliers for customer locations
# There are a lot of customers that are outside the "vendor bubble"
# vendor bubble == customers that aren't outliers

lat_lo, lat_hi = -25, 25
long_lo, long_hi = -5, 5
c_outliers = (train_locations['latitude'] < lat_lo) | (train_locations['latitude'] > lat_hi) | (train_locations['longitude'] < long_lo) | (train_locations['longitude'] > long_hi)
v_outliers = (vendors['latitude'] < lat_lo) | (vendors['latitude'] > lat_hi) | (vendors['longitude'] < long_lo) | (vendors['longitude'] > long_hi)

# Want to transform outliers so that they are closer to vendors, but also stay in their clusters
# Project outliers onto ellipse around bubble

lat_radius = lat_hi
long_radius = long_hi

# Project customer outliers
for i in tqdm(train_locations[c_outliers].index):
        lat = train_locations.loc[i, 'latitude']
        long = train_locations.loc[i, 'longitude']
        mag = sqrt(lat**2 + long**2)
        train_locations.loc[i, 'latitude'] = lat / mag * lat_radius
        train_locations.loc[i, 'longitude'] = long / mag * long_radius

# Project vendor outliers
for i in tqdm(vendors[v_outliers].index):
        lat = vendors.loc[i, 'latitude']
        long = vendors.loc[i, 'longitude']
        mag = sqrt(lat**2 + long**2)
        vendors.loc[i, 'latitude'] = lat / mag * lat_radius
        vendors.loc[i, 'longitude'] = long / mag * long_radius
```

```python colab={"base_uri": "https://localhost:8080/", "height": 350} id="9_tk5ZRpWWtO" executionInfo={"status": "ok", "timestamp": 1628704529655, "user_tz": -330, "elapsed": 1315, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="639f5ef3-2837-4533-e830-e8fea3d3e28d"
plt.figure(figsize=(6, 5))
plt.scatter(x=train_locations.longitude, y=train_locations.latitude, label='Customers', marker='s', alpha=0.2)
plt.scatter(x=vendors.longitude, y=vendors.latitude, label='Vendors', marker='*', alpha=0.5, s=vendors['num_orders']/5, c=vendors['avg_sale'], cmap='plasma')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='lower right')
plt.colorbar(label='$ Avg Sale')
plt.title('Customer + Vendor Locations (Outliers Transformed)')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 357} id="LxYkphxVWkly" executionInfo={"status": "ok", "timestamp": 1628704573226, "user_tz": -330, "elapsed": 690, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c993a4a4-19e9-430c-d533-5c2d937d491b"
# drop some columns
keep_continuous = ['latitude', 'longitude', 'serving_distance', 'prepration_time', 'vendor_rating', 'num_orders_log3', 'amt_sales_log3', 'avg_sale_log']
keep_categorical = ['vendor_category_id', 'delivery_charge', 'status', 'rank', 'primary_tags', 'vendor_tag']
keep_columns = keep_continuous + keep_categorical
vendors = vendors[keep_columns]
vendors.head()
```

```python id="PZzXvnYYYOQn"
# encode categorical features
vendors, _, _ = integer_encoding(df=vendors, cols=['vendor_category_id', 'delivery_charge', 'status', 'rank', 'primary_tags'], drop_old=True, monotone_mapping=True)
vendors = multiclass_list_encoding(df=vendors, cols=['primary_tags', 'vendor_tag'], drop_old=True)

# convert to tensor
# first row is zeros to act as "null token" for customer sequences
vendors_tensor = torch.cat((torch.zeros([1, len(vendors.columns)]), torch.tensor(vendors.values)), axis=0)

# sort orders by datetime
train_orders['created_at'] = pd.to_datetime(train_orders['created_at'])
train_orders.sort_values(by=['created_at'], inplace=True)
# test_orders['created_at'] = pd.to_datetime(test_orders['created_at'])
# test_orders.sort_values(by=['created_at'], inplace=True)
orders_grp = train_orders.groupby(by=['customer_id'])

# map vendor ids to range(1, num_vendors+1)
train_orders['vendor_id'] = train_orders['vendor_id'].map(v_id_map)
# test_orders['vendor_id'] = test_orders['vendor_id'].map(v_id_map)

# group sequences by customer_id
train_sequences = get_sequences(df=train_orders, target='vendor_id', group_by=['customer_id'])
train_lengths = train_sequences.apply(len).value_counts(normalize=True).sort_index()
# test_sequences = get_sequences(df=test_orders, target='vendor_id', group_by=['customer_id'])
# test_lengths = test_sequences.apply(len).value_counts(normalize=True).sort_index()

# get padded sequences
window = 6
train_sequences_padded = pandas_sequences_to_tensor(sequences=train_sequences, window=window)

# custom pytorch dataset
train_sequences_padded_dataset = CustomDataset(train_sequences_padded)
```

<!-- #region id="yN13bE4SYHVq" -->
## Save Sequence Datasets and Vendors Tensor
<!-- #endregion -->

```python id="YF8Dkk3QXzym" executionInfo={"status": "ok", "timestamp": 1628706194369, "user_tz": -330, "elapsed": 391, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
scaler = StandardScaler()

vendors_tensor = torch.tensor(scaler.fit_transform(vendors_tensor))

with open("./data/silver/train_sequences_padded_dataset.pkl", "wb") as file:
    pickle.dump(train_sequences_padded_dataset, file)

with open("./data/silver/vendors_tensor.pkl", "wb") as file:
    pickle.dump(vendors_tensor, file)
```
