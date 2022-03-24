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

<!-- #region id="ex0dBumEm1tE" -->
# IEEE Challenge 2021 Data Preprocessing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="joAvJDL8hIie" executionInfo={"status": "ok", "timestamp": 1637734912898, "user_tz": -330, "elapsed": 5136, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="df2fbfae-6a7d-4baf-fa26-f28839d2c8d0"
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/train.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/item_info.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track1_testset.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track2_testset.parquet.snappy
```

```python id="gjC-ZhlLhNsw"
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
```

```python id="pBd13g-zm24j"
pd.set_option('display.max_columns', None) 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="dTbK9JfyfBc8" executionInfo={"status": "ok", "timestamp": 1637735002099, "user_tz": -330, "elapsed": 1010, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="62c5a0c2-e711-47d4-e5d3-397099400f44"
df_train = pd.read_parquet('train.parquet.snappy')
df_train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="DG2EyaSNgr4p" executionInfo={"status": "ok", "timestamp": 1637735008046, "user_tz": -330, "elapsed": 417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d412e84d-dce4-429a-8ec1-19b47c059a99"
df_item_info = pd.read_parquet('item_info.parquet.snappy')
df_item_info.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="MaNCX7AcfGNy" executionInfo={"status": "ok", "timestamp": 1637735009395, "user_tz": -330, "elapsed": 702, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1b17ed02-e16f-40c0-e9ff-58097dd0003b"
df_test1 = pd.read_parquet('track1_testset.parquet.snappy')
df_test1.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="89GEPUJEfOMi" executionInfo={"status": "ok", "timestamp": 1637735010860, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f8389471-ab6d-474d-ba01-f206d15740ab"
df_test2 = pd.read_parquet('track2_testset.parquet.snappy')
df_test2.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="JqapzW5NikBs" executionInfo={"status": "ok", "timestamp": 1637296655297, "user_tz": -330, "elapsed": 11461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="793260e9-6106-42c6-a5ba-c754f1524909"
portraitidx_to_idx_dict_list = []
for i in range(10):
    portraitidx_to_idx_dict_list.append(dict())
acculumated_idx = [0] * 10

for i in tqdm(range(df_train.shape[0])):
    user_portrait = [int(s) for s in df_train.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
            acculumated_idx[idx] += 1

for i in tqdm(range(df_test1.shape[0])):
    user_portrait = [int(s) for s in df_test1.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
            acculumated_idx[idx] += 1

for i in tqdm(range(df_test2.shape[0])):
    user_portrait = [int(s) for s in df_test2.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
            acculumated_idx[idx] += 1
```

```python colab={"base_uri": "https://localhost:8080/"} id="L9-OrnvQe9sh" executionInfo={"status": "ok", "timestamp": 1637296771983, "user_tz": -330, "elapsed": 603, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fe3239b9-9789-4835-dd89-017aabb289c4"
acculumated_idx
```

```python colab={"base_uri": "https://localhost:8080/"} id="KFJorv84e9n2" executionInfo={"status": "ok", "timestamp": 1637296811536, "user_tz": -330, "elapsed": 629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="31413693-2cfe-4cc9-b0b0-aac5188aa217"
portraitidx_to_idx_dict_list[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="BGYI1zFNe9lI" executionInfo={"status": "ok", "timestamp": 1637297036786, "user_tz": -330, "elapsed": 726, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a494a10d-9d98-4a8d-fb62-6be3bc75e9e7"
dict(list(portraitidx_to_idx_dict_list[1].items())[0:10])
```

```python colab={"base_uri": "https://localhost:8080/"} id="veBgZ6TphBoN" executionInfo={"status": "ok", "timestamp": 1637297267504, "user_tz": -330, "elapsed": 842, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9be3e2a7-fdab-4ec3-cda8-57bb534a5028"
# item info
item_info_dict = {}
for i in tqdm(range(df_item_info.shape[0])):
    item_id = df_item_info.at[i, 'item_id'] 

    item_discrete = df_item_info.at[i, 'item_vec'].split(',')[:3]
    item_cont = df_item_info.at[i, 'item_vec'].split(',')[-2:]
    price = df_item_info.at[i, 'price'] / 3000
    loc = df_item_info.at[i, 'location'] - 1 # 0~2

    item_cont.append(price) # 2 + 1
    item_discrete.append(loc) # 3 + 1

    item_cont = [float(it) for it in item_cont]
    item_discrete = [int(it) for it in item_discrete]
    item_discrete[0] = item_discrete[0] - 1 # 1~4 -> 0~3
    item_discrete[2] = item_discrete[2] - 1 # 1~2 -> 0~1

    item_info_dict[int(item_id)] = {
        'cont': np.array(item_cont, dtype=np.float64),
        'discrete': np.array(item_discrete, dtype=np.int64),
    }
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="G8Ar3dV-hC-Z" executionInfo={"status": "ok", "timestamp": 1637297181268, "user_tz": -330, "elapsed": 727, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="23ed07b2-ab4c-46c8-93b7-c3aa53889320"
df_item_info.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="bNObz0X4haGM" executionInfo={"status": "ok", "timestamp": 1637297316627, "user_tz": -330, "elapsed": 1114, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="53a836e7-a13e-43f8-ce7f-0024f1e00e61"
dict(list(item_info_dict.items())[0:10])
```

```python id="1tBra0avi68f" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637297736063, "user_tz": -330, "elapsed": 33962, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d93aec13-5752-4d72-d37c-dbd353bb2545"
# trainset
train_samples = []
val_samples = []

# shuffle
# df_train = shuffle(df_train, random_state=2333).reset_index() # not shuffling - for this tutorial
total_num = int(df_train.shape[0])
num_train = int(total_num * 0.95)
num_val = total_num - num_train

for i in tqdm(range(total_num)):
    if df_train.at[i, 'user_click_history'] == '0:0':
        user_click_list = [0]
    else:
        user_click_list = df_train.at[i, 'user_click_history'].split(',')
        user_click_list = [int(sample.split(':')[0]) for sample in user_click_list]
    num_user_click_history = len(user_click_list)
    tmp = np.zeros(400, dtype=np.int64)
    tmp[:len(user_click_list)] = user_click_list
    user_click_list = tmp
    
    exposed_items = [int(s) for s in df_train.at[i, 'exposed_items'].split(',')]
    labels = [int(s) for s in df_train.at[i, 'labels'].split(',')]

    user_portrait = [int(s) for s in df_train.at[i, 'user_protrait'].split(',')]
    # portraitidx_to_idx_dict_list: list of 10 dict, int:int
    for j in range(10):
        user_portrait[j] = portraitidx_to_idx_dict_list[j][user_portrait[j]]
    for k in range(9):
        one_sample = {
            'user_click_list': user_click_list,
            'num_user_click_history': num_user_click_history,
            'user_portrait': np.array(user_portrait, dtype=np.int64),
            'item_id': exposed_items[k],
            'label': labels[k]
        }
        if i < num_train:
            train_samples.append(one_sample)
        else:
            val_samples.append(one_sample)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="6k0Jj0uriSSo" executionInfo={"status": "ok", "timestamp": 1637297504540, "user_tz": -330, "elapsed": 625, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="22830205-9562-44c5-c821-87ee99dc0fb5"
df_train.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="mFxlyhTajE8R" executionInfo={"status": "ok", "timestamp": 1637297738254, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2438a83d-1cf0-4df5-b7de-3e3921081ace"
train_samples[0]
```

```python id="4tb4yDY4jaWW"
class BigDataCupDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 item_info_dict,
                 database
                ):
        super().__init__()
        self.item_info_dict = item_info_dict
        self.database = database

    def __len__(self, ):
        return len(self.database)

    def __getitem__(self, idx):
        one_sample = self.database[idx]
        user_click_history = one_sample['user_click_list']
        num_user_click_history = one_sample['num_user_click_history']
        user_discrete_feature = one_sample['user_portrait']
        item_id = one_sample['item_id']
        item_discrete_feature = self.item_info_dict[item_id]['discrete']
        item_cont_feature = self.item_info_dict[item_id]['cont']
        label = one_sample['label']

        # print(num_user_click_history)

        user_click_history = torch.IntTensor(user_click_history)
        num_user_click_history = torch.IntTensor([num_user_click_history])
        user_discrete_feature = torch.IntTensor(user_discrete_feature)
        item_id = torch.IntTensor([item_id])
        item_discrete_feature = torch.IntTensor(item_discrete_feature)
        item_cont_feature = torch.FloatTensor(item_cont_feature)
        label = torch.IntTensor([label])

        # print(num_user_click_history)

        return user_click_history, num_user_click_history, user_discrete_feature, \
               item_id, item_discrete_feature, item_cont_feature, label
```

```python id="IVrctleBjo9Y"
train_ds = BigDataCupDataset(item_info_dict, train_samples)
```

```python colab={"base_uri": "https://localhost:8080/"} id="0NbhWM4Rj70n" executionInfo={"status": "ok", "timestamp": 1637297985815, "user_tz": -330, "elapsed": 989, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bf321263-8c45-4ddb-e91f-e26eb10f35f5"
for i in range(len(train_ds)):
    sample = train_ds[i]
    print(sample)
    if i == 1:
        break
```

```python colab={"base_uri": "https://localhost:8080/"} id="sYDIMZaJjrfD" executionInfo={"status": "ok", "timestamp": 1637298026792, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a1b6a742-4411-47a8-db2a-79303bc1ce56"
train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
train_dl
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="tMg11oR-khrv" executionInfo={"status": "ok", "timestamp": 1637298090921, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5a85537b-8c5b-473b-9c59-d7811b487930"
df_test1.head()
```

```python id="pmE37O6yjhqc" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637298497018, "user_tz": -330, "elapsed": 25418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3b1587df-999d-4c1d-b35d-6044aa51ccf1"
# testset
test_samples = []

# shuffle
total_num = int(df_test1.shape[0])

for i in tqdm(range(total_num)):
    if df_test1.at[i, 'user_click_history'] == '0:0':
        user_click_list = [0]
    else:
        user_click_list = df_test1.at[i, 'user_click_history'].split(',')
        user_click_list = [int(sample.split(':')[0]) for sample in user_click_list]
    num_user_click_history = len(user_click_list)
    tmp = np.zeros(400, dtype=np.int64)
    tmp[:len(user_click_list)] = user_click_list
    user_click_list = tmp
    
    exposed_items = [int(s) for s in df_test1.at[i, 'exposed_items'].split(',')]
    labels = [int(s) for s in df_test1.at[i, 'labels'].split(',')]

    user_portrait = [int(s) for s in df_test1.at[i, 'user_protrait'].split(',')]
    # portraitidx_to_idx_dict_list: list of 10 dict, int:int
    for j in range(10):
        user_portrait[j] = portraitidx_to_idx_dict_list[j][user_portrait[j]]
    for k in range(9):
        one_sample = {
            'user_click_list': user_click_list,
            'num_user_click_history': num_user_click_history,
            'user_portrait': np.array(user_portrait, dtype=np.int64),
            'item_id': exposed_items[k],
        }
        test_samples.append(one_sample)
```

```python colab={"base_uri": "https://localhost:8080/"} id="eyFR5cpYkszD" executionInfo={"status": "ok", "timestamp": 1637298497020, "user_tz": -330, "elapsed": 40, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5d80a480-4cc6-4eee-bf62-6b0cfe58ee9a"
test_samples[0]
```

```python id="gujtFyHPj71n"
class BigDataCupTestDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 item_info_dict,
                 database
                ):
        super().__init__()
        self.item_info_dict = item_info_dict
        self.database = database

    def __len__(self, ):
        return len(self.database)

    def __getitem__(self, idx):
        one_sample = self.database[idx]
        user_click_history = one_sample['user_click_list']
        num_user_click_history = one_sample['num_user_click_history']
        user_discrete_feature = one_sample['user_portrait']
        item_id = one_sample['item_id']
        item_discrete_feature = self.item_info_dict[item_id]['discrete']
        item_cont_feature = self.item_info_dict[item_id]['cont']

        user_click_history = torch.IntTensor(user_click_history)
        num_user_click_history = torch.IntTensor([num_user_click_history])
        user_discrete_feature = torch.IntTensor(user_discrete_feature)
        item_id = torch.IntTensor([item_id])
        item_discrete_feature = torch.IntTensor(item_discrete_feature)
        item_cont_feature = torch.FloatTensor(item_cont_feature)

        return user_click_history, num_user_click_history, user_discrete_feature, \
               item_id, item_discrete_feature, item_cont_feature
```

```python colab={"base_uri": "https://localhost:8080/"} id="K1UNzz7okGPo" executionInfo={"status": "ok", "timestamp": 1637298551296, "user_tz": -330, "elapsed": 617, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3f3c922a-3e08-4158-c548-cd4a720f3fe1"
val_ds = BigDataCupTestDataset(item_info_dict, test_samples)

for i in range(len(val_ds)):
    sample = val_ds[i]
    print(sample)
    if i == 1:
        break
```

```python colab={"base_uri": "https://localhost:8080/"} id="RzuFh7dskzfZ" executionInfo={"status": "ok", "timestamp": 1637298566961, "user_tz": -330, "elapsed": 921, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e5f1b60c-e1a2-4c70-9098-7a762cd5b9bd"
val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=9, shuffle=False)
val_dl
```
