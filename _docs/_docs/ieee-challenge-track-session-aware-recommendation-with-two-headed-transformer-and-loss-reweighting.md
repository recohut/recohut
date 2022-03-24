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

<!-- #region id="n9xp6JfMkspy" -->
We will implement multitask model - buy and click.

Model 1:
- Input
 - 9 products displayed to users (emb, discrete attributes, continuous attributes of products)
 - User's click history (emb, discrete attributes, continuous attributes of products)
 - User attributes (discrete attributes)
 - User purchase time (month, day, day of the week, hour)
- Output
 - The user purchased the first session (purchased 0-3, 4-6, 7-9, three types of sessions)
 - Whether the user bought these 9 products (you can use the 4 types of product reweighting loss mentioned by Gaochen)
 
Model 2:
- Input
 - User’s previous click history (commodity emb, discrete attributes, and continuous attributes become discrete)
 - The product currently clicked by the user (the emb, discrete attributes, and continuous attributes of the product become discrete)
 - User attributes (discrete attributes)
- Output
 - Whether the user clicked on this product

The above two models share all emb.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="joAvJDL8hIie" executionInfo={"status": "ok", "timestamp": 1637146618868, "user_tz": -330, "elapsed": 8500, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9423712b-58be-4eb9-ec86-bb78fd7761fd"
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/train.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/item_info.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track1_testset.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track2_testset.parquet.snappy
```

```python id="yDqafKgoxCw8"
%load_ext tensorboard
# %reload_ext tensorboard
%tensorboard --logdir=./
```

<!-- #region id="mcNWsLo0q1zg" -->
## pre
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="597GI-kjCYuj" executionInfo={"status": "ok", "timestamp": 1629797423629, "user_tz": -480, "elapsed": 4239, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="73a8b8ea-8baa-4647-ea5e-90dc69233b16"
!pip install einops
```

```python id="K4cozZgaq-CJ"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import datetime as dt
from datetime import datetime, timezone
tz_bj = dt.timezone(dt.timedelta(hours=8))

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import einops
```

<!-- #region id="W_8YS5mOjuMx" -->
## time EDA
<!-- #endregion -->

```python id="IZff7VyIjwj-"
data_path='/content/'
df_train = pd.read_parquet(f'{data_path}/trainset.parquet.snappy')
df_test1 = pd.read_parquet(f'{data_path}/track1_testset.parquet.snappy')
```

```python colab={"base_uri": "https://localhost:8080/"} id="TiSiA92ujwhB" executionInfo={"status": "ok", "timestamp": 1629797433689, "user_tz": -480, "elapsed": 16, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="2a950971-9a58-4e6c-b549-8c48f29e4e0c"
df_train['time']
```

```python colab={"base_uri": "https://localhost:8080/"} id="dUrRZYhnkrxX" executionInfo={"status": "ok", "timestamp": 1629797433690, "user_tz": -480, "elapsed": 12, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="7821638f-dd7a-47ba-d892-d7265831a259"
df_test1['time']
```

```python colab={"base_uri": "https://localhost:8080/"} id="nt5CAWh5jweY" executionInfo={"status": "ok", "timestamp": 1629797434123, "user_tz": -480, "elapsed": 443, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="2ffd32f5-3998-4f0d-95ba-9c19a63e77e9"
plt.hist(df_train['time'], color='blue', edgecolor='black', bins=120)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="g620XQKJkvO3" executionInfo={"status": "ok", "timestamp": 1629797434664, "user_tz": -480, "elapsed": 549, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="f56440ba-cd3f-495e-d455-17f037ba29a1"
plt.hist(df_test1['time'], color='blue', edgecolor='black', bins=120)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="i_7dHkbfjwWo" executionInfo={"status": "ok", "timestamp": 1629797434671, "user_tz": -480, "elapsed": 32, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="e9045439-617b-4839-a75b-c19ba70a5339"
import datetime as dt
from datetime import datetime, timezone
tz_bj = dt.timezone(dt.timedelta(hours=8))

print(datetime.utcfromtimestamp(1582992009).strftime('%Y-%m-%d %H:%M:%S'))
print(datetime.utcfromtimestamp(1593014357).strftime('%Y-%m-%d %H:%M:%S'))

print(datetime.fromtimestamp(1582992009, tz_bj).strftime('%Y-%m-%d %H:%M:%S'))
print(datetime.fromtimestamp(1593014357, tz_bj).strftime('%Y-%m-%d %H:%M:%S'))

a = datetime.fromtimestamp(1593014357, tz_bj)
month   = int(a.strftime('%m')) - 1 # 几月 01, 02, …, 12 (-1)
date    = int(a.strftime('%d')) - 1 # 几号 01, 02, …, 31 (-1)
weekday = int(a.strftime('%w')) # 星期几 0, 1, …, 6. 0:sunday, 6:saturday
hour    = int(a.strftime('%H')) # 小时 00, 01, …, 23

print(month, date, weekday, hour)
```

```python id="wVg41xh3jwHW"

```

<!-- #region id="2gYQ0Kd4TLf9" -->
## 把所有的 feature 改成 离散的分桶
<!-- #endregion -->

```python id="esis3V3iTSAV"
def load_item_info(data_path='/content/'):
    # item info
    df_item_info = pd.read_parquet(f'{data_path}/item_info.parquet.snappy')
    item_info_dict = {}
    for i in tqdm(range(df_item_info.shape[0])):
        item_id = df_item_info.at[i, 'item_id'] 

        item_discrete = df_item_info.at[i, 'item_vec'].split(',')[:3]
        item_cont = df_item_info.at[i, 'item_vec'].split(',')[-2:]
        price = df_item_info.at[i, 'price'] # / 3000
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
    return item_info_dict
```

```python colab={"base_uri": "https://localhost:8080/"} id="fkoc0I3oTchv" executionInfo={"status": "ok", "timestamp": 1629797435456, "user_tz": -480, "elapsed": 809, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="87b194ac-a7fe-4ac3-c74c-72cdfdafe250"
item_info_dict = load_item_info(data_path='/content/')
```

```python colab={"base_uri": "https://localhost:8080/"} id="hpjwCe0oTjgx" executionInfo={"status": "ok", "timestamp": 1629797435457, "user_tz": -480, "elapsed": 42, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="5cb3fa5e-aa16-42c2-d786-2586f0437e60"
item_info_dict[1]
```

```python id="n75khv80Tjdk"
cont1, cont2, cont3 = [], [], []
for k, v in item_info_dict.items():
    c = v['cont']
    cont1.append(c[0])
    cont2.append(c[1])
    cont3.append(c[2])
cont1, cont2, cont3 = np.array(cont1), np.array(cont2), np.array(cont3)
```

<!-- #region id="0f38lqoAdjme" -->
### item cont1
<!-- #endregion -->

```python id="r6o_S5snAbO_"
cont1_nonzero = cont1[cont1 != 0]
cont1_nonzero_log = np.log(cont1_nonzero)
```

```python colab={"base_uri": "https://localhost:8080/"} cellView="code" id="ke_-93M1Wzxw" executionInfo={"status": "ok", "timestamp": 1629797435462, "user_tz": -480, "elapsed": 38, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="58d67e1b-323b-491c-d49d-36100bf85716"
fig1, ax1 = plt.subplots()
# ax1.set_title('Basic Plot')
ax1.boxplot(cont1)
plt.show()

plt.hist(cont1, color='blue', edgecolor='black', bins=20)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="E4m5n7_zAYLY" executionInfo={"status": "ok", "timestamp": 1629797435920, "user_tz": -480, "elapsed": 492, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="5ca79c99-f7bc-4cff-fc7b-928761b063d1"
fig1, ax1 = plt.subplots()
# ax1.set_title('Basic Plot')
ax1.boxplot(cont1_nonzero[cont1_nonzero < 5])
plt.show()

plt.hist(cont1_nonzero[cont1_nonzero < 5], color='blue', edgecolor='black', bins=100)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="MKAATkz8A0tF" executionInfo={"status": "ok", "timestamp": 1629797435931, "user_tz": -480, "elapsed": 33, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="e2a768ee-1c95-452a-82d6-0bb8a8cd4b40"
fig1, ax1 = plt.subplots()
# ax1.set_title('Basic Plot')
ax1.boxplot(cont1_nonzero_log[cont1_nonzero_log < 0])
plt.show()

plt.hist(cont1_nonzero_log[cont1_nonzero_log < 0], color='blue', edgecolor='black', bins=3)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="unT58aTZA0ni" executionInfo={"status": "ok", "timestamp": 1629797436183, "user_tz": -480, "elapsed": 281, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="42c575ce-b532-49ba-9dd3-aa8c5fcbbef8"
def cont1_to_discrete(cont_val):
    if cont_val == 0:
        return 0
    cont_val_log = np.log(cont_val)
    if cont_val_log > 0:
        return 4
    if cont_val_log < -7.5:
        return 1
    if cont_val_log < -5.5:
        return 2
    if cont_val_log < 0:
        return 3

cont1_discrete = []
for c in cont1:
    tmp = cont1_to_discrete(c)
    cont1_discrete.append(tmp)

plt.hist(cont1_discrete, color='blue', edgecolor='black', bins=100)
plt.show()
```

```python id="kjmFTo7QTjI7"

```

<!-- #region id="ovZbrp7vDg7t" -->
### item cont2
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SZSyy6aNDjQd" executionInfo={"status": "ok", "timestamp": 1629797436619, "user_tz": -480, "elapsed": 442, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="0c889560-f0d5-4bef-edcb-42bb7557fcf2"
fig1, ax1 = plt.subplots()
# ax1.set_title('Basic Plot')
ax1.boxplot(cont2[cont2 != 0])
plt.show()

plt.hist(cont2[cont2 != 0], color='blue', edgecolor='black', bins=9)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="uWJVZgQFDjND" executionInfo={"status": "ok", "timestamp": 1629797437062, "user_tz": -480, "elapsed": 455, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="5a993fa5-2094-4106-f9c2-3ce1bec4cc8b"
def cont2_to_discrete(cont_val):
    if cont_val == 0:
        return 0
    if cont_val < 0.1:
        return 1
    if cont_val < 0.2:
        return 2
    if cont_val < 0.3:
        return 3
    if cont_val < 0.4:
        return 4
    if cont_val < 0.5:
        return 5
    if cont_val < 0.6:
        return 6
    if cont_val < 0.7:
        return 7
    if cont_val < 0.8:
        return 8
    if cont_val < 0.9:
        return 9

cont2_discrete = []
for c in cont2:
    tmp = cont2_to_discrete(c)
    cont2_discrete.append(tmp)

plt.hist(cont2_discrete, color='blue', edgecolor='black', bins=100)
plt.show()
```

<!-- #region id="-7USelB6EjOt" -->
### item cont3
<!-- #endregion -->

```python id="EzM7U7DIErSs"
cont3_log = np.log(cont3)
```

```python colab={"base_uri": "https://localhost:8080/"} id="IcEYUzS2DjKl" executionInfo={"status": "ok", "timestamp": 1629797437502, "user_tz": -480, "elapsed": 452, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="68f1ad90-fd31-441d-e4f7-e176f0dfe98f"
fig1, ax1 = plt.subplots()
# ax1.set_title('Basic Plot')
ax1.boxplot(cont3)
plt.show()

plt.hist(cont3, color='blue', edgecolor='black', bins=9)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="AJA5UgQjFGRz" executionInfo={"status": "ok", "timestamp": 1629797438106, "user_tz": -480, "elapsed": 623, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="8732158e-28db-4ae8-f974-4ca74178300b"
fig1, ax1 = plt.subplots()
# ax1.set_title('Basic Plot')
ax1.boxplot(cont3[cont3 < 5000])
plt.show()

plt.figure(figsize=(10, 3))
plt.hist(cont3[cont3 < 5000], color='blue', edgecolor='black', bins=20)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="uZnEqwdgFP_O" executionInfo={"status": "ok", "timestamp": 1629797438108, "user_tz": -480, "elapsed": 12, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="f9caa1f4-7663-4c4d-ccc0-d5e0e77e712b"
fig1, ax1 = plt.subplots()
# ax1.set_title('Basic Plot')
ax1.boxplot(cont3[cont3 >= 2000])
plt.show()

plt.hist(cont3[cont3 >= 2000], color='blue', edgecolor='black', bins=9)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="sYOgk3kQDi8x" executionInfo={"status": "ok", "timestamp": 1629797438649, "user_tz": -480, "elapsed": 550, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="b082c941-3bbd-4f4b-a03c-e8c226d82253"
## price
def cont3_to_discrete(cont_val):
    if cont_val < 300:
        return 0
    if cont_val < 500:
        return 1
    if cont_val < 750:
        return 2
    if cont_val < 1000:
        return 3
    if cont_val < 1500:
        return 4
    if cont_val < 2000:
        return 5
    if cont_val < 2500:
        return 6
    if cont_val < 3000:
        return 7
    if cont_val < 3500:
        return 8
    if cont_val <= 5000:
        return 9
    if cont_val > 5000:
        return 10

cont3_discrete = []
for c in cont3:
    tmp = cont3_to_discrete(c)
    cont3_discrete.append(tmp)

plt.hist(cont3_discrete, color='blue', edgecolor='black', bins=100)
plt.show()
```

```python id="SP2ioZXKE82U"

```

```python id="9ZhgU9BEE8yp"

```

<!-- #region id="RHcX9ocwGzFL" -->
## overall data cont to discrete

cont1,2,3 -> discrete 

len = [5, 10, 11]
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Dtz3cxCAHAhl" executionInfo={"status": "ok", "timestamp": 1629797438945, "user_tz": -480, "elapsed": 306, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="cdaf9873-5ec9-46a5-a785-fee8688c4776"
def cont1_to_discrete(cont_val):
    if cont_val == 0:
        return 0
    cont_val_log = np.log(cont_val)
    if cont_val_log < -7.5:
        return 1
    if cont_val_log < -5.5:
        return 2
    if cont_val_log <= 0:
        return 3
    if cont_val_log > 0:
        return 4

cont1_discrete = []
for c in cont1:
    tmp = cont1_to_discrete(c)
    cont1_discrete.append(tmp)
assert len(cont1_discrete) == len(cont1)

plt.hist(cont1_discrete, color='blue', edgecolor='black', bins=100)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="BlGNyFmOG9AZ" executionInfo={"status": "ok", "timestamp": 1629797438947, "user_tz": -480, "elapsed": 17, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="0682405f-5c0a-4255-a817-f27fdb33558b"
def cont2_to_discrete(cont_val):
    if cont_val == 0:
        return 0
    if cont_val < 0.1:
        return 1
    if cont_val < 0.2:
        return 2
    if cont_val < 0.3:
        return 3
    if cont_val < 0.4:
        return 4
    if cont_val < 0.5:
        return 5
    if cont_val < 0.6:
        return 6
    if cont_val < 0.7:
        return 7
    if cont_val < 0.8:
        return 8
    if cont_val < 0.9:
        return 9

cont2_discrete = []
for c in cont2:
    tmp = cont2_to_discrete(c)
    cont2_discrete.append(tmp)
assert len(cont2_discrete) == len(cont2)

plt.hist(cont2_discrete, color='blue', edgecolor='black', bins=100)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="f-0rcghEE8uR" executionInfo={"status": "ok", "timestamp": 1629797439686, "user_tz": -480, "elapsed": 404, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="08a723bb-ae7f-4d7d-ff93-466ce5c29b49"
## price
def cont3_to_discrete(cont_val):
    if cont_val < 300:
        return 0
    if cont_val < 500:
        return 1
    if cont_val < 750:
        return 2
    if cont_val < 1000:
        return 3
    if cont_val < 1500:
        return 4
    if cont_val < 2000:
        return 5
    if cont_val < 2500:
        return 6
    if cont_val < 3000:
        return 7
    if cont_val < 3500:
        return 8
    if cont_val <= 5000:
        return 9
    if cont_val > 5000:
        return 10

cont3_discrete = []
for c in cont3:
    tmp = cont3_to_discrete(c)
    cont3_discrete.append(tmp)
assert len(cont3_discrete) == len(cont3)

plt.hist(cont3_discrete, color='blue', edgecolor='black', bins=100)
plt.show()
```

```python id="VlGhfVuXE8q2"
def load_item_info_turn_cont_to_discrete(
    data_path='/content/'
):

    # item info
    df_item_info = pd.read_parquet(f'{data_path}/item_info.parquet.snappy')

    num_items = 381+1 # 0 means no item; normal items start from 1
    num_features = (3+1) + (2+1)
    item_features = np.zeros((num_items, num_features)).astype(np.int64)

    for i in tqdm(range(num_items - 1)):
        item_id = df_item_info.at[i, 'item_id']
        # discrete
        item_discrete = df_item_info.at[i, 'item_vec'].split(',')[:3]
        loc = df_item_info.at[i, 'location'] - 1 # 0~2
        item_discrete.append(loc)
        item_discrete = [int(it) for it in item_discrete]
        item_discrete[0] = item_discrete[0] - 1 # 1~4 -> 0~3
        item_discrete[2] = item_discrete[2] - 1 # 1~2 -> 0~1

        # cont
        item_cont = df_item_info.at[i, 'item_vec'].split(',')[-2:]
        price = df_item_info.at[i, 'price']
        item_cont.append(price)
        item_cont = [float(it) for it in item_cont]

        item_cont1 = cont1_to_discrete(item_cont[0])
        item_cont2 = cont2_to_discrete(item_cont[1])
        item_cont3 = cont3_to_discrete(item_cont[2])

        # agg
        item_discrete.append(item_cont1)
        item_discrete.append(item_cont2)
        item_discrete.append(item_cont3)

        item_total_feat = np.array(item_discrete, dtype=np.int64)
        item_features[item_id] = item_total_feat
    
    # change 0 item to no-feature (last idx of each feature + 1)
    last_idx = np.max(item_features, axis=0)
    item_features[0] = last_idx + 1

    return item_features
```

```python colab={"base_uri": "https://localhost:8080/"} id="z0G1Ui7qGyFN" executionInfo={"status": "ok", "timestamp": 1629797439688, "user_tz": -480, "elapsed": 21, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="d8806ea6-10e1-443b-b87b-1db45d6d9b91"
item_features = load_item_info_turn_cont_to_discrete(
    data_path='/content/'
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="LmdZpqNLLRit" executionInfo={"status": "ok", "timestamp": 1629797439689, "user_tz": -480, "elapsed": 18, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="3a7998b4-2d97-4e40-93c0-43e75b4175c2"
print(item_features[:10])
```

<!-- #region id="XLUSRSm_q0m6" -->
## data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eZO24jMGrxur" executionInfo={"status": "ok", "timestamp": 1629797455390, "user_tz": -480, "elapsed": 15717, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="0fcdef19-7f15-4a93-b969-a71e79faa844"
## 获取 user portrait 的映射，因为
data_path='/content/'
# portraitidx_to_idx_dict_list: list of 10 dict, int:int

portraitidx_to_idx_dict_list = []
for i in range(10):
    portraitidx_to_idx_dict_list.append(dict())
acculumated_idx = [0] * 10


df_train = pd.read_parquet(f'{data_path}/trainset.parquet.snappy')
for i in tqdm(range(df_train.shape[0])):
    user_portrait = [int(s) for s in df_train.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
            acculumated_idx[idx] += 1
print(acculumated_idx)


# 测试集中如果出现训练集里没出现的， 就统一置为最后一个
df_test1 = pd.read_parquet(f'{data_path}/track1_testset.parquet.snappy')
for i in tqdm(range(df_test1.shape[0])):
    user_portrait = [int(s) for s in df_test1.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
df_test2 = pd.read_parquet(f'{data_path}/track2_testset.parquet.snappy')
for i in tqdm(range(df_test2.shape[0])):
    user_portrait = [int(s) for s in df_test2.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]

for i in range(10):
    acculumated_idx[i] += 1

# 所以最后也统一加上一个， 即使有些维度其实没有 测试集出现但训练集没出现的东西
print(acculumated_idx)
```

```python id="WvWsdWAP1t00"
def load_train_data(data_path='/content/'):
    # trainset
    train_samples = []
    val_samples = []
    df_train = pd.read_parquet(f'{data_path}/trainset.parquet.snappy')

    # shuffle
    df_train = shuffle(df_train, random_state=2333).reset_index()
    total_num = int(df_train.shape[0])
    num_train = int(total_num * 0.95)
    num_val = total_num - num_train # 5% validation data

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
        one_sample = {
            'user_click_list': user_click_list,
            'num_user_click_history': num_user_click_history,
            'user_portrait': np.array(user_portrait, dtype=np.int64),
            'item_id': np.array(exposed_items, dtype=np.int64),
            'label': np.array(labels, dtype=np.int64),
            't': df_train.at[i, 'time'] # int
        }
        if i < num_train:
            train_samples.append(one_sample)
        else:
            val_samples.append(one_sample)
    return train_samples, val_samples
```

```python colab={"base_uri": "https://localhost:8080/"} id="GaHYRMrE4oV5" executionInfo={"status": "ok", "timestamp": 1629797482454, "user_tz": -480, "elapsed": 27075, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="e5ea35b2-cc50-4608-be5a-87222c19e236"
train_samples, val_samples = load_train_data(data_path='/content/')
```

```python id="LhnPDrwEq8GK"
# aug items within sess
from itertools import permutations
from functools import reduce
import operator
import random

perm1 = list(permutations([0, 1, 2]))
perm2 = list(permutations([3, 4, 5]))
perm3 = list(permutations([6, 7, 8]))

aug_order = []
for p1 in perm1:
    # print(p1)
    for p2 in perm2:
        # print(p1, p2)
        for p3 in perm3:
            # print(p1, p2, p3)
            tmp = reduce(operator.concat, [p1, p2, p3])
            aug_order.append(tmp)
len_aug_order = len(aug_order)


class BigDataCupDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 item_features,
                 database,
                 get_click_data=True,
                 train_val='train' # if train, use augorder
                ):
        super().__init__()
        self.item_features = item_features
        self.database = database
        self.train_val = train_val
        self.get_click_data = get_click_data

    def __len__(self, ):
        return len(self.database)

    def __getitem__(self, idx):
        one_sample = self.database[idx]
        user_click_history = one_sample['user_click_list'] # [400]
        num_user_click_history = one_sample['num_user_click_history'] # int
        user_discrete_feature = one_sample['user_portrait'] # [10]
        nine_item_id = one_sample['item_id'] # [9]
        label = one_sample['label'] # [9]
        t = one_sample['t'] # int

        a = datetime.fromtimestamp(t, tz_bj)
        month   = int(a.strftime('%m')) - 1 # 几月 01, 02, …, 12 (-1)
        date    = int(a.strftime('%d')) - 1 # 几号 01, 02, …, 31 (-1)
        weekday = int(a.strftime('%w')) # 星期几 0, 1, …, 6. 0:sunday, 6:saturday
        hour    = int(a.strftime('%H')) # 小时 00, 01, …, 23

        if self.train_val == 'train':
            ao = list(aug_order[random.randint(0, len_aug_order - 1)])
            nine_item_id = nine_item_id[ao]
            label = label[ao]

        user_click_history_discrete_feature = np.zeros((400, (3+1) + (2+1))).astype(np.int64)
        for i in range(num_user_click_history):
            if user_click_history[i] == 0:
                user_click_history_discrete_feature[i] = self.item_features[user_click_history[i]]
                # 这里 0表示没有任何点击
            else:
                user_click_history_discrete_feature[i] = self.item_features[user_click_history[i]]

        nine_item_discrete_feature = np.zeros((9, (3+1) + (2+1))).astype(np.int64)
        for i in range(9):
            nine_item_discrete_feature[i] = self.item_features[nine_item_id[i]]

        session_label = 0 # 0,1,2,3
        # 0: 什么都不买
        for i in range(9):
            if label[i]: # 买1~3个
                session_label = 1
            if i >= 3 and label[i]: # 买4~6个
                session_label = 2
            if i >= 6 and label[i]: # 买7~9个
                session_label = 3

        # click
        if self.get_click_data:
            def neg_sample(): # 这里没有考虑到 buy 和 click，但就先随机吧
                return random.randint(1, 381)

            click_user_discrete_feature = user_discrete_feature
            click_user_click_history = user_click_history
            click_user_click_history_discrete_feature = user_click_history_discrete_feature
            if num_user_click_history == 1:
                click_user_click_history = user_click_history
                click_num_user_click_history = num_user_click_history
                click_item_id = neg_sample() # random sample (todo)
                click_item_discrete_feature = torch.IntTensor(self.item_features[click_item_id])
                click_label = torch.IntTensor([0])
            else: # num_user_click_history >= 2
                # random sample to a click history thre
                click_idx = random.randint(2, num_user_click_history) # 要预测的那个点击item
                click_num_user_click_history = click_idx - 1 # 预测的点击item之前有多少东西
                # pos or neg 1:4
                if random.randint(1, 3) == 1:
                    # pos
                    click_item_id = click_user_click_history[click_idx - 1]
                    click_label = torch.IntTensor([1])
                else:
                    # neg
                    click_item_id = neg_sample()
                    click_label = torch.IntTensor([0])
                click_item_discrete_feature = torch.IntTensor(self.item_features[click_item_id])

        # buy
        user_click_history = torch.IntTensor(user_click_history)
        user_click_history_discrete_feature = torch.IntTensor(user_click_history_discrete_feature)
        num_user_click_history = torch.IntTensor([num_user_click_history])
        user_discrete_feature = torch.IntTensor(user_discrete_feature)
        nine_item_id = torch.IntTensor(nine_item_id)
        nine_item_discrete_feature = torch.IntTensor(nine_item_discrete_feature)
        label = torch.IntTensor(label)
        session_label = session_label

        if not self.get_click_data:
            return user_click_history, \
                user_click_history_discrete_feature, \
                num_user_click_history, \
                nine_item_id, \
                nine_item_discrete_feature, \
                user_discrete_feature, \
                label, session_label, \
                month, date, weekday, hour
        else:
            # click
            click_user_click_history = torch.IntTensor(user_click_history)
            click_user_click_history_discrete_feature = torch.IntTensor(click_user_click_history_discrete_feature)
            click_num_user_click_history = torch.IntTensor([click_num_user_click_history])
            click_item_id = torch.IntTensor([click_item_id])
            click_item_discrete_feature = torch.IntTensor(click_item_discrete_feature)
            click_user_discrete_feature = torch.IntTensor(click_user_discrete_feature)
            click_label = torch.IntTensor([click_label])

            return user_click_history, \
            user_click_history_discrete_feature, \
            num_user_click_history, \
            nine_item_id, \
            nine_item_discrete_feature, \
            user_discrete_feature, \
            label, session_label, \
            month, date, weekday, hour, \
            click_user_click_history, \
            click_user_click_history_discrete_feature, \
            click_num_user_click_history, \
            click_item_id, \
            click_item_discrete_feature, \
            click_user_discrete_feature, \
            click_label

```

```python id="GeMu5TTrsW4y"
ds = BigDataCupDataset(item_features, train_samples, get_click_data=True, train_val='train')
```

```python colab={"base_uri": "https://localhost:8080/"} id="C43dO07JNsZu" executionInfo={"status": "ok", "timestamp": 1629797482960, "user_tz": -480, "elapsed": 13, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="1136a70a-f37b-49e5-b687-3f0e56cfd2c9"
ds[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="O4gRymk3y4NA" executionInfo={"status": "ok", "timestamp": 1629797510283, "user_tz": -480, "elapsed": 27334, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="56ab13ba-df75-4916-f5dd-2180e66b83f5"
train_samples, val_samples = load_train_data()

train_ds = BigDataCupDataset(item_features, train_samples, get_click_data=True, train_val='train')
train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=32, shuffle=True)

val_ds = BigDataCupDataset(item_features, val_samples, get_click_data=True, train_val='val')
val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=32, shuffle=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="YLzEpwfVuCwK" executionInfo={"status": "ok", "timestamp": 1629797510679, "user_tz": -480, "elapsed": 417, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="28d6b576-c164-4b9c-d972-0a71ae21d24c"
next(iter(train_dl))
```

<!-- #region id="iI3v-E1gq3HQ" -->
## model
<!-- #endregion -->

<!-- #region id="WUHvb6uj377U" -->
### transformer
<!-- #endregion -->

```python id="kvu-Sv9urDIo"
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 hidden_size,
                 qkv_size, 
                 num_heads, 
                 dropout_ratio=0.
                ):
        super().__init__()
        self.n = num_heads
        self.d = qkv_size
        self.D = hidden_size

        self.scale = self.d ** -0.5
        self.to_qkv = nn.Linear(self.D, self.n * self.d * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(self.n * self.d, self.D),
            nn.Dropout(dropout_ratio)
        )
        
    def forward(self, x):
        """
        x: BND
        output: BND
        """
        B, N, D = x.shape
        
        # get qkv
        qkv_agg = self.to_qkv(x) # BND -> BN(num_heads*qkv_size*3)
        qkv_agg = qkv_agg.chunk(3, dim=-1) # BND -> 3 * [BN(num_heads*qkv_size)]
        q = einops.rearrange(qkv_agg[0], 'B N (n d) -> B n N d', n=self.n)
        k = einops.rearrange(qkv_agg[1], 'B N (n d) -> B n N d', n=self.n)
        v = einops.rearrange(qkv_agg[2], 'B N (n d) -> B n N d', n=self.n)

        # calc self attention 
        dots = torch.einsum('Bnid, Bnjd -> Bnij', q, k)     # BnNd, BnNd -> BnNN
        attn = self.attend(dots * self.scale)
        out = torch.einsum('BnNj, Bnjd -> BnNd', attn, v)   # BnNN, BnNd -> BnNd
        out = einops.rearrange(out, 'B n N d -> B N (n d)') # BnNd -> BN(nd) = BND

        # aggregate multihead
        out = self.to_out(out)
        
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, 
                 hidden_size,
                 mlp_size,
                 dropout_ratio
                ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(mlp_size, hidden_size),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        """
        x: BND
        output: BND
        """
        return self.model(x)

```

```python id="olI3IXdfyOZ2"
class MultitaskTransformer(nn.Module):
    def __init__(self, 
                 num_items=381,
                 hidden_size=128,
                 num_layers=3, 
                 mlp_size=64, # normally = 4 * hidden_size
                 qkv_size=32, # normally = 64 = hidden_size / num_heads
                 num_heads=4, 
                 msa_dropout_ratio=0.1, 
                 ffn_dropout_ratio=0.1, 
                 device='cpu'
                ):
        """
        除了 item_emb 之外，其余的 emb 编号都是 0 开始的
        """

        super().__init__()
        self.device = device
        self.num_items = num_items
        self.NUM_ITEM_DISCRETE_FEATURE = 3+1 + 2+1 # item_vec3+location1 + item_vec2+price1
        self.NUM_USER_DISCRETE_FEATURE = 10
        self.hidden_size = hidden_size
        self.N_buy = 1 + self.NUM_ITEM_DISCRETE_FEATURE + \
                     9 * (1 + self.NUM_ITEM_DISCRETE_FEATURE) + \
                     self.NUM_USER_DISCRETE_FEATURE + \
                     4 # time feat
        self.N_click = 1 + self.NUM_ITEM_DISCRETE_FEATURE + \
                       1 + self.NUM_ITEM_DISCRETE_FEATURE + \
                       self.NUM_USER_DISCRETE_FEATURE

        # item emb
        self.item_emb = nn.Embedding(self.num_items + 1, self.hidden_size) # 0 表示没有记录，因此 num_items + 1

        # item discrete feature
        self.item_discrete_feature_emb_list = nn.ModuleList()
        num_unique_value_list = [4+1, 10+1, 2+1, 3+1, 5+1, 10+1, 11+1] # [4, 10, 2, 3]
        for i in range(self.NUM_ITEM_DISCRETE_FEATURE):
            num_unique_value = num_unique_value_list[i]
            self.item_discrete_feature_emb_list.append(
                nn.Embedding(num_unique_value, self.hidden_size)
            )
        
        # user discrete feature
        self.user_discrete_feature_emb_list = nn.ModuleList()
        num_unique_value_list = [4, 1364, 21, 11, 196, 50, 4, 12, 3, 2165] # (already add 1 for features in test but not in train)
        for i in range(self.NUM_USER_DISCRETE_FEATURE):
            num_unique_value = num_unique_value_list[i]
            self.user_discrete_feature_emb_list.append(
                nn.Embedding(num_unique_value, self.hidden_size)
            )
        
        # position emb
        self.position_emb_buy   = nn.Parameter(torch.randn(1, self.N_buy, self.hidden_size))
        self.position_emb_click = nn.Parameter(torch.randn(1, self.N_click, self.hidden_size))

        # time emb
        self.month_emb   = nn.Embedding(12, self.hidden_size)
        self.date_emb    = nn.Embedding(31, self.hidden_size)
        self.weekday_emb = nn.Embedding(7,  self.hidden_size)
        self.hour_emb    = nn.Embedding(24, self.hidden_size)

        # month, date, weekday, hour
        # month   = int(a.strftime('%m')) - 1 # 几月 01, 02, …, 12 (-1)
        # date    = int(a.strftime('%d')) - 1 # 几号 01, 02, …, 31 (-1)
        # weekday = int(a.strftime('%w')) # 星期几 0, 1, …, 6. 0:sunday, 6:saturday
        # hour    = int(a.strftime('%H')) # 小时 00, 01, …, 23


        # transformer layers
        self.transformer_layers_buy = nn.ModuleList([])
        for _ in range(num_layers):
            self.transformer_layers_buy.append(nn.ModuleList([
                nn.Sequential( # MSA(LN(x))
                    nn.LayerNorm(self.hidden_size),
                    MultiHeadSelfAttention(self.hidden_size, qkv_size, num_heads, msa_dropout_ratio),
                ),
                nn.Sequential( # MLPs(LN(x))
                    nn.LayerNorm(self.hidden_size),
                    FeedForwardNetwork(self.hidden_size, mlp_size, ffn_dropout_ratio)
                )
            ]))
        self.transformer_layers_click = nn.ModuleList([])
        for _ in range(num_layers):
            self.transformer_layers_click.append(nn.ModuleList([
                nn.Sequential( # MSA(LN(x))
                    nn.LayerNorm(self.hidden_size),
                    MultiHeadSelfAttention(self.hidden_size, qkv_size, num_heads, msa_dropout_ratio),
                ),
                nn.Sequential( # MLPs(LN(x))
                    nn.LayerNorm(self.hidden_size),
                    FeedForwardNetwork(self.hidden_size, mlp_size, ffn_dropout_ratio)
                )
            ]))

        # session prediction head
        self.session_prediction_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.PReLU(),
            nn.Linear(64, 4)
        )

        # buy prediction head
        self.buy_prediction_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.PReLU(),
            nn.Linear(64, 9)
        )

        # click prediction head
        self.click_prediction_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.PReLU(),
            nn.Linear(64, 1)
        )


    def get_item_emb_attr(self, 
                          item_id, 
                          item_discrete_feature):
        """
        param:
            item_id:               [B, 9]   (0表示没有记录，从1开始是真的item)
            item_discrete_feature: [B, 9, NUM_USER_DISCRETE_FEATURE]
        return: 
            emb_attr:
                [B(batchsize), 9, N(num_feat=1+7), D(hiddendim)]
        note: 
            above, 9 can be an arbitrary number, e.g. 400
        """
        tmp = []
        # item emb
        item_emb = self.item_emb(item_id) # [B, 9, D]
        tmp.append(torch.unsqueeze(item_emb, 2)) # [B, 9, 1, D]
        # item discrete feature emb
        for i in range(self.NUM_ITEM_DISCRETE_FEATURE):
            a = self.item_discrete_feature_emb_list[i](item_discrete_feature[:, :, i]) # [B, 9, D]
            tmp.append(torch.unsqueeze(a, 2)) # [B, 9, 1, D]
        # cat to [B, 9, N, D]
        return torch.cat(tmp, dim=2) # [B, 9, 8, D]

    def forward(self,
                user_click_history, user_click_history_discrete_feature, num_user_click_history,
                nine_item_id, nine_item_discrete_feature,
                user_discrete_feature,
                month, date, weekday, hour
                ):
        """
        用户的点击历史记录（商品的emb、离散属性）
            user_click_history: [B, 400], 最多有400个点击历史记录, 每个里面是itemid, 0表示没有记录
            user_click_history_discrete_feature: [B, 400, 3+1 + 2+1]
            num_user_click_history: [B, 1], 用户点击历史数量
        展示给用户的9个商品（商品的emb、离散属性、连续属性）
            nine_item_id: [B, 9], 商品id
            nine_item_discrete_feature: [B, 9, 3+1 + 2+1] 商品离散属性（已重映射） item_vec3 + location1 + item_vec2 + price1
        用户的属性（离散属性）
            user_discrete_feature: [B, 10] 用户离散属性（已重映射）
        时间 feat：
            month, date, weekday, hour [B]
        """

        batch_size = user_click_history.size()[0]

        # 用户的点击历史记录（商品的emb、离散属性）
        user_click_history_emb = torch.zeros( # [B, 8, D]
            (batch_size, 1 + self.NUM_ITEM_DISCRETE_FEATURE, self.hidden_size)
        ).to(self.device)
        assert 1 + self.NUM_ITEM_DISCRETE_FEATURE == 8
        tmp = self.get_item_emb_attr(user_click_history, user_click_history_discrete_feature) # [B, 400, 8, D]
        for i in range(batch_size):
            aa = tmp[i, :num_user_click_history[i], :, :] # [B, 400, 8, D] -> [400-, 8, D]
            a = torch.mean(aa, dim=0) # [400-, 8, D] -> [8, D]
            user_click_history_emb[i] = a
        
        # 展示给用户的9个商品（商品的emb、离散属性）
        nine_item_emb = self.get_item_emb_attr(nine_item_id, nine_item_discrete_feature) # [B, 9, 8, D]
        nine_item_emb = einops.rearrange(nine_item_emb, 'B n N D -> B (n N) D') # [B, 9*8, D]

        # 用户的属性（离散属性）
        tmp = []
        for i in range(self.NUM_USER_DISCRETE_FEATURE):
            a = self.user_discrete_feature_emb_list[i](user_discrete_feature[:, i]) # [B, D]
            tmp.append(torch.unsqueeze(a, 1)) # [B, 1, D]
        user_discrete_feature_emb = torch.cat(tmp, dim=1) # [B, 10, D]

        # time feat: month, date, weekday, hour
        m_emb = torch.unsqueeze(self.month_emb(month), 1) # [B, D]
        d_emb = torch.unsqueeze(self.date_emb(date), 1)
        w_emb = torch.unsqueeze(self.weekday_emb(weekday), 1)
        h_emb = torch.unsqueeze(self.hour_emb(hour), 1)

        # concat all emb
        z0 = torch.cat([user_click_history_emb,     # [B, 8, D]
                        nine_item_emb,              # [B, 9*8, D]
                        user_discrete_feature_emb,  # [B, 10, D]
                        m_emb, d_emb, w_emb, h_emb  # [B, 1, D] * 4
                        ], dim=1) # [B, N, D]

        position_embs = einops.repeat(self.position_emb_buy, '() N D -> B N D', B=batch_size)
        z0 = z0 + position_embs

        # transformer
        zl = z0
        for transformer_layer in self.transformer_layers_buy:
            zl = zl + transformer_layer[0](zl) # MSA(LN(x))
            zl = zl + transformer_layer[1](zl) # MLPs(LN(x))

        # global average pooling
        zl = einops.reduce(zl, 'B N D -> B D', reduction='mean')

        # head
        session_pred = self.session_prediction_head(zl)
        buy_pred = self.buy_prediction_head(zl)

        return session_pred, buy_pred # [B, 4], [B, 9]


    def forward_click(self,
                      user_click_history, user_click_history_discrete_feature, num_user_click_history,
                      item_id, item_discrete_feature,
                      user_discrete_feature):
        """
        用户 之前的 点击历史记录（商品的emb、离散属性、连续属性变成离散）
            user_click_history: [N, 400], 最多有400个点击历史记录, 每个里面是itemid, 0表示没有记录
            user_click_history_discrete_feature: [N, 400, 3+1 + 2+1]
            num_user_click_history: [N, 1], 用户点击历史数量
        用户 __当前点击__ 的商品（商品的emb、离散属性、连续属性变成离散）
            item_id: [N, 1], 商品id
            item_discrete_feature: [N, 3+1 + 2+1] 商品离散属性（已重映射） item_vec3 + location1 + item_vec2 + price1
        用户的属性（离散属性）
            user_discrete_feature: [B, 10] 用户离散属性（已重映射）

        输出：
        1. 用户是否点击这个商品
        
        """
        batch_size = user_click_history.size()[0]

        # 用户的点击历史记录（商品的emb、离散属性）
        user_click_history_emb = torch.zeros( # [B, 7+1, D]
            (batch_size, 1 + self.NUM_ITEM_DISCRETE_FEATURE, self.hidden_size)
        ).to(self.device)
        assert 1 + self.NUM_ITEM_DISCRETE_FEATURE == 8
        # print(user_click_history.device, user_click_history_discrete_feature.device, flush=True)
        tmp = self.get_item_emb_attr(user_click_history, user_click_history_discrete_feature) # [B, 400, 8, D]
        for i in range(batch_size):
            aa = tmp[i, :num_user_click_history[i], :, :] # [B, 400, 8, D] -> [400-, 8, D]
            a = torch.mean(aa, dim=0) # [400-, 8, D] -> [8, D]
            user_click_history_emb[i] = a
        
        # 用户 __当前点击__ 的商品（商品的emb、离散属性）
        item_discrete_feature = torch.unsqueeze(item_discrete_feature, dim=1) # [B, 7] -> [B, 1, 7]
        # print(item_id.shape, item_discrete_feature.shape)
        item_emb = self.get_item_emb_attr(item_id, item_discrete_feature) # [B, 1, 8, D]
        item_emb = einops.rearrange(item_emb, 'B n N D -> B (n N) D') # [B, 1*8, D]

        # 用户的属性（离散属性）
        tmp = []
        for i in range(self.NUM_USER_DISCRETE_FEATURE):
            a = self.user_discrete_feature_emb_list[i](user_discrete_feature[:, i]) # [B, D]
            tmp.append(torch.unsqueeze(a, 1)) # [B, 1, D]
        user_discrete_feature_emb = torch.cat(tmp, dim=1) # [B, 10, D]

        # concat all emb
        z0 = torch.cat([user_click_history_emb,     # [B, 8, D]
                        item_emb,              # [B, 1*8, D]
                        user_discrete_feature_emb,  # [B, 10, D]
                        ], dim=1) # [B, N, D]

        position_embs = einops.repeat(self.position_emb_click, '() N D -> B N D', B=batch_size)
        z0 = z0 + position_embs

        # transformer
        zl = z0
        for transformer_layer in self.transformer_layers_click:
            zl = zl + transformer_layer[0](zl) # MSA(LN(x))
            zl = zl + transformer_layer[1](zl) # MLPs(LN(x))

        # global average pooling
        zl = einops.reduce(zl, 'B N D -> B D', reduction='mean')

        # head
        click_pred = self.click_prediction_head(zl)

        return click_pred # [B, 1]

```

```python colab={"base_uri": "https://localhost:8080/"} id="737_d9n124ld" executionInfo={"status": "ok", "timestamp": 1629797517589, "user_tz": -480, "elapsed": 6634, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="4d94efe5-7e2e-4555-876f-66e168fcbf27"
m = MultitaskTransformer(
    num_items=381,
    hidden_size=128,
    num_layers=3, 
    mlp_size=64, # normally = 4 * hidden_size
    qkv_size=32, # normally = 64 = hidden_size / num_heads
    num_heads=4, 
    msa_dropout_ratio=0.1, 
    ffn_dropout_ratio=0.1, 
    device='cuda'
)
m = m.to('cuda')

B = 3
a = m(
    user_click_history=torch.ones([B, 400], dtype=torch.int32).cuda(),
    user_click_history_discrete_feature=torch.ones([B, 400, 7], dtype=torch.int32).cuda(),
    num_user_click_history=torch.ones([B, 1], dtype=torch.int32).cuda() * 10,
    user_discrete_feature=torch.ones([B, 10], dtype=torch.int32).cuda(),
    nine_item_id=torch.ones([B, 9], dtype=torch.int32).cuda(),
    nine_item_discrete_feature=torch.ones([B, 9, 7], dtype=torch.int32).cuda(),
    month=torch.zeros([B], dtype=torch.int32).cuda(),
    date=torch.zeros([B], dtype=torch.int32).cuda(), 
    weekday=torch.zeros([B], dtype=torch.int32).cuda(), 
    hour=torch.zeros([B], dtype=torch.int32).cuda()
)
print(a)

b = m.forward_click(
    user_click_history=torch.ones([B, 400], dtype=torch.int32).cuda(),
    user_click_history_discrete_feature=torch.ones([B, 400, 7], dtype=torch.int32).cuda(),
    num_user_click_history=torch.ones([B, 1], dtype=torch.int32).cuda() * 10,
    user_discrete_feature=torch.ones([B, 10], dtype=torch.int32).cuda(),
    item_id=torch.ones([B, 1], dtype=torch.int32).cuda(),
    item_discrete_feature=torch.ones([B, 7], dtype=torch.int32).cuda()
)
print(b)
```

<!-- #region id="TLla7fQWACja" -->
## train
<!-- #endregion -->

```python id="RXFOYhEYx0hL"
model_name = 'multitask_transformer_augorder_reweight_timefeat_adamlr0.001_epoch10'
tb_path = 'runs/%s-%s' % (datetime.today().strftime('%Y-%m-%d-%H:%M:%S'), model_name)
tb_writer = SummaryWriter(tb_path)
```

```python id="p0bNqBiJzRI1"
device = 'cuda'
model = MultitaskTransformer(
    num_items=381,
    hidden_size=128,
    num_layers=3, 
    mlp_size=64, # normally = 4 * hidden_size
    qkv_size=32, # normally = 64 = hidden_size / num_heads
    num_heads=4, 
    msa_dropout_ratio=0.1, 
    ffn_dropout_ratio=0.1, 
    device='cuda'
)
model = model.to(device)
```

```python id="1p5BR29K3GzG"
## load
tb_path = 'runs/2021-08-21-15:51:42-multitask_transformer_augorder_reweight_adamlr0.001_epoch10'
tb_writer = SummaryWriter(tb_path)


device ='cuda'
model = torch.load(f'{tb_path}/model_epoch10.pth', map_location=device)
model = model.to(device)
```

```python id="SgJASuk0Zh4y"
def binary_acc(sess_pred, y_pred, y_test):
    # print(sess_pred)
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    y_pred_tag_intact = y_pred_tag.clone()


    ##################################
    ## vanilla
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc1 = correct_results_sum / y_test.shape[0] / 9

    real_acc1 = 0.0
    for i in range(y_test.shape[0]):
        correct_results_sum = (y_pred_tag[i] == y_test[i]).sum().float()
        one_acc = correct_results_sum / 9
        if one_acc == 1:
            real_acc1 += 1
    real_acc1 = real_acc1 / y_test.shape[0]
    # print(y_pred_tag)

    ####################################
    ## use sess to refine y_pred_tag
    for i in range(y_test.shape[0]):
        if sess_pred[i] == 0:
            y_pred_tag[i][:] = 0
        elif sess_pred[i] == 1:
            y_pred_tag[i][3:] = 0
        elif sess_pred[i] == 2:
            y_pred_tag[i][:3] = 1
            y_pred_tag[i][6:] = 0
        elif sess_pred[i] == 3:
            y_pred_tag[i][:6] = 1

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc2 = correct_results_sum / y_test.shape[0] / 9

    real_acc2 = 0.0
    for i in range(y_test.shape[0]):
        correct_results_sum = (y_pred_tag[i] == y_test[i]).sum().float()
        one_acc = correct_results_sum / 9
        if one_acc == 1:
            real_acc2 += 1
    real_acc2 = real_acc2 / y_test.shape[0]
    # print(y_pred_tag)

    #######################################
    ## rule 2
    y_pred_tag = y_pred_tag_intact
    acc_rule2 = 0.0
    real_acc_rule2 = 0.0
    for i in range(y_test.shape[0]):
        for j in range(9):
            k = 8 - j
            if k >= 6 and y_pred_tag[i][k] == 1:
                y_pred_tag[i][:6] = 1
            if k >= 3 and y_pred_tag[i][k] == 1:
                y_pred_tag[i][:3] = 1
        
        correct_results_sum = (y_pred_tag[i] == y_test[i]).sum().float()
        a = correct_results_sum / 9
        acc_rule2 += a
        if a == 1:
            real_acc_rule2 += 1
    acc_rule2 = acc_rule2 / y_test.shape[0]
    real_acc_rule2 = real_acc_rule2 / y_test.shape[0]
    # print(y_pred_tag)


    return acc1, acc2, acc_rule2, real_acc1, real_acc2, real_acc_rule2
```

```python id="zi18rA3KxvAQ"
def click_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc
```

```python id="N3eOxzVAHhr9"
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

```python id="KveZBH9APprY"
NUM_EPOCH = 10

batches_done = 0
best_val_acc = 0
```

```python id="XqAKhqym4OnZ"
batches_done = 7722 * 10
best_val_acc = 0.38792
```

```python id="qLYwI5xr9aAF"
sess_criterion = nn.CrossEntropyLoss()
buy_criterion = nn.BCEWithLogitsLoss(reduction='none')
click_criterion = nn.BCEWithLogitsLoss()
```

```python id="5uh3Bd3EASid" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629635114582, "user_tz": -480, "elapsed": 5043630, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="c4085db4-d0af-498a-f3ad-6830a75437f9"
# for epoch_idx in range(NUM_EPOCH):  # loop over the dataset multiple times
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
for epoch_idx in range(10, 15):  # loop over the dataset multiple times

    train_running_sess_loss = 0.0
    train_running_buy_loss = 0.0
    train_running_click_loss = 0.0
    train_cnt = 0
    train_click_acc = 0
    train_sess_acc_sum = 0
    train_buy_acc1_sum = 0
    train_buy_real_acc1_sum = 0
    train_buy_acc2_sum = 0
    train_buy_real_acc2_sum = 0
    train_buy_acc_rule2_sum = 0
    train_buy_real_acc_rule2_sum = 0
    train_cnt_session_0 = train_cnt_session_1 = train_cnt_session_2 = train_cnt_session_3 = 0
    
    for i, data in enumerate(train_dl, 0):
        model.train()

        # get the inputs; data is a list of [inputs, labels]
        user_click_history, \
            user_click_history_discrete_feature, \
            num_user_click_history, \
            item_id, item_discrete_feature, \
            user_discrete_feature, label, session_label, \
            month, date, weekday, hour, \
            click_user_click_history, \
            click_user_click_history_discrete_feature, \
            click_num_user_click_history, \
            click_item_id, \
            click_item_discrete_feature, \
            click_user_discrete_feature, \
            click_label = data
        
        train_batch_size = user_click_history.shape[0]
        
        user_click_history = user_click_history.to(device)
        user_click_history_discrete_feature = user_click_history_discrete_feature.to(device)
        num_user_click_history = num_user_click_history.to(device)
        item_id = item_id.to(device)
        item_discrete_feature = item_discrete_feature.to(device)
        user_discrete_feature = user_discrete_feature.to(device)
        label = label.to(device)
        session_label = session_label.to(device)
        month = month.to(device)
        date = date.to(device)
        weekday = weekday.to(device)
        hour = hour.to(device)

        click_user_click_history = click_user_click_history.to(device)
        click_user_click_history_discrete_feature = click_user_click_history_discrete_feature.to(device)
        click_num_user_click_history = click_num_user_click_history.to(device)
        click_item_id = click_item_id.to(device)
        click_item_discrete_feature = click_item_discrete_feature.to(device)
        click_user_discrete_feature = click_user_discrete_feature.to(device)
        click_label = click_label.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        sess_outputs, buy_outputs = model(
            user_click_history,
            user_click_history_discrete_feature,
            num_user_click_history,
            item_id,
            item_discrete_feature,
            user_discrete_feature,
            month, date, weekday, hour
        )
        click_outputs = model.forward_click(
            click_user_click_history,
            click_user_click_history_discrete_feature,
            click_num_user_click_history,
            click_item_id,
            click_item_discrete_feature,
            click_user_discrete_feature
        )

        sess_loss = sess_criterion(sess_outputs, session_label)
        buy_loss = buy_criterion(buy_outputs, label.float()) # [N, 9]
        for b_idx in range(train_batch_size):
            if session_label[b_idx] == 0:
                w = torch.Tensor([0.5,0.5,0.5, 0.5,0.5,0.5, 0.5,0.5,0.5]).to(device)
            elif session_label[b_idx] == 1:
                w = torch.Tensor([1,1,1, 0.5,0.5,0.5, 0.5,0.5,0.5]).to(device)
            elif session_label[b_idx] == 2:
                w = torch.Tensor([0.5,0.5,0.5, 1,1,1, 0.5,0.5,0.5]).to(device)
            elif session_label[b_idx] == 3:
                w = torch.Tensor([0.5,0.5,0.5, 0.5,0.5,0.5, 1,1,1]).to(device)
            buy_loss[b_idx] = buy_loss[b_idx] * w
        buy_loss = torch.mean(buy_loss)
        click_loss = click_criterion(click_outputs, click_label.float())
        loss = 0.1 * sess_loss + 0.8 * buy_loss + 0.1 * click_loss
        # loss = click_loss

        loss.backward()
        optimizer.step()

        # print statistics
        train_running_sess_loss += sess_loss.item()
        train_running_buy_loss += buy_loss.item()
        train_running_click_loss += click_loss.item()

        _, sess_predicted = torch.max(sess_outputs.data, 1)
        sess_acc = (sess_predicted == session_label).sum().item() / train_batch_size
        # buy_acc1, buy_acc2, buy_real_acc1, buy_real_acc2 = binary_acc(sess_predicted, buy_outputs, label)
        buy_acc1, buy_acc2, buy_acc_rule2, buy_real_acc1, buy_real_acc2, buy_real_acc_rule2 = binary_acc(sess_predicted, buy_outputs, label)

        train_click_acc += click_acc(click_outputs, click_label)

        train_sess_acc_sum += sess_acc
        train_buy_acc1_sum += buy_acc1
        train_buy_real_acc1_sum += buy_real_acc1
        train_buy_acc2_sum += buy_acc2
        train_buy_real_acc2_sum += buy_real_acc2
        train_buy_acc_rule2_sum += buy_acc_rule2
        train_buy_real_acc_rule2_sum += buy_real_acc_rule2
        train_cnt += 1

        # train_cnt_session_0 += torch.sum(session_label == 0)
        # train_cnt_session_1 += torch.sum(session_label == 1)
        # train_cnt_session_2 += torch.sum(session_label == 2)
        # train_cnt_session_3 += torch.sum(session_label == 3)

        batches_done += 1

        if i % 50 == 1:
            print(i, end=' ')

        if i % 500 == 1 and i != 1:    # print every 2000 mini-batches
            print('----- TRAIN -----')
            print('[%d, %5d] sess loss: %.3f' % (epoch_idx + 1, i + 1, train_running_sess_loss / train_cnt))
            print('[%d, %5d] buy loss: %.3f'  % (epoch_idx + 1, i + 1, train_running_buy_loss / train_cnt))
            print('[%d, %5d] click loss: %.3f'  % (epoch_idx + 1, i + 1, train_running_click_loss / train_cnt))
            print('- sess acc:',      train_sess_acc_sum / train_cnt, flush=True)
            print('- buy acc1:',      train_buy_acc1_sum.cpu().item() / train_cnt, flush=True)
            print('- buy real acc1:', train_buy_real_acc1_sum / train_cnt, flush=True)
            print('- buy acc2:',      train_buy_acc2_sum.cpu().item() / train_cnt, flush=True)
            print('- buy real acc2:', train_buy_real_acc2_sum / train_cnt, flush=True)
            print('- buy acc rule2:', train_buy_acc_rule2_sum.cpu().item() / train_cnt, flush=True)
            print('- buy real acc rule2:', train_buy_real_acc_rule2_sum / train_cnt, flush=True)
            print('- click acc:', train_click_acc / train_cnt, flush=True)
            # print('- train sess cnt:', train_cnt_session_0, train_cnt_session_1, train_cnt_session_2, train_cnt_session_3)

            tb_writer.add_scalar('train/sess loss', train_running_sess_loss / train_cnt, batches_done)
            tb_writer.add_scalar('train/buy loss',  train_running_buy_loss / train_cnt, batches_done)
            tb_writer.add_scalar('train/click loss',  train_running_click_loss / train_cnt, batches_done)
            tb_writer.add_scalar('train/sess acc',  train_sess_acc_sum / train_cnt, batches_done)
            tb_writer.add_scalar('train/buy acc1',  train_buy_acc1_sum.cpu().item() / train_cnt, batches_done)
            tb_writer.add_scalar('train/buy real acc1', train_buy_real_acc1_sum / train_cnt, batches_done)
            tb_writer.add_scalar('train/buy acc2',      train_buy_acc2_sum.cpu().item() / train_cnt, batches_done)
            tb_writer.add_scalar('train/buy real acc2', train_buy_real_acc2_sum / train_cnt, batches_done)
            tb_writer.add_scalar('train/buy acc rule2', train_buy_acc_rule2_sum.cpu().item() / train_cnt, batches_done)
            tb_writer.add_scalar('train/buy real acc rule2', train_buy_real_acc_rule2_sum / train_cnt, batches_done)
            tb_writer.add_scalar('train/click acc',  train_click_acc / train_cnt, batches_done)
            
            train_running_sess_loss = 0.0
            train_running_buy_loss = 0.0
            train_running_click_loss = 0.0
            train_cnt = 0
            train_click_acc = 0
            train_sess_acc_sum = 0
            train_buy_acc1_sum = 0
            train_buy_real_acc1_sum = 0
            train_buy_acc2_sum = 0
            train_buy_real_acc2_sum = 0
            train_buy_acc_rule2_sum = 0
            train_buy_real_acc_rule2_sum = 0
            train_cnt_session_0 = train_cnt_session_1 = train_cnt_session_2 = train_cnt_session_3 = 0

            ## val
            model.eval()
            valid_running_sess_loss = 0.0
            valid_running_buy_loss = 0.0
            valid_running_click_loss = 0.0
            valid_cnt = 0
            valid_click_acc = 0
            valid_sess_acc_sum = 0
            valid_buy_acc1_sum = 0
            valid_buy_real_acc1_sum = 0
            valid_buy_acc2_sum = 0
            valid_buy_real_acc2_sum = 0
            valid_buy_acc_rule2_sum = 0
            valid_buy_real_acc_rule2_sum = 0
            
            valid_cnt_session_0 = valid_cnt_session_1 = valid_cnt_session_2 = valid_cnt_session_3 = 0

            for _, val_data in tqdm(enumerate(val_dl, 0)):
                user_click_history, \
                    user_click_history_discrete_feature, \
                    num_user_click_history, \
                    item_id, item_discrete_feature, \
                    user_discrete_feature, label, session_label, \
                    month, date, weekday, hour, \
                    click_user_click_history, \
                    click_user_click_history_discrete_feature, \
                    click_num_user_click_history, \
                    click_item_id, \
                    click_item_discrete_feature, \
                    click_user_discrete_feature, \
                    click_label = val_data

                valid_batch_size = user_click_history.shape[0]

                user_click_history = user_click_history.to(device)
                user_click_history_discrete_feature = user_click_history_discrete_feature.to(device)
                num_user_click_history = num_user_click_history.to(device)
                item_id = item_id.to(device)
                item_discrete_feature = item_discrete_feature.to(device)
                user_discrete_feature = user_discrete_feature.to(device)
                label = label.to(device)
                session_label = session_label.to(device)

                month = month.to(device)
                date = date.to(device)
                weekday = weekday.to(device)
                hour = hour.to(device)

                click_user_click_history = click_user_click_history.to(device)
                click_user_click_history_discrete_feature = click_user_click_history_discrete_feature.to(device)
                click_num_user_click_history = click_num_user_click_history.to(device)
                click_item_id = click_item_id.to(device)
                click_item_discrete_feature = click_item_discrete_feature.to(device)
                click_user_discrete_feature = click_user_discrete_feature.to(device)
                click_label = click_label.to(device)


                sess_outputs, buy_outputs = model(
                    user_click_history,
                    user_click_history_discrete_feature,
                    num_user_click_history,
                    item_id,
                    item_discrete_feature,
                    user_discrete_feature, 
                    month, date, weekday, hour
                )
                click_outputs = model.forward_click(
                    click_user_click_history,
                    click_user_click_history_discrete_feature,
                    click_num_user_click_history,
                    click_item_id,
                    click_item_discrete_feature,
                    click_user_discrete_feature
                )

                sess_loss = sess_criterion(sess_outputs, session_label)
                buy_loss = buy_criterion(buy_outputs, label.float())
                for b_idx in range(valid_batch_size):
                    if session_label[b_idx] == 0:
                        w = torch.Tensor([0.5,0.5,0.5, 0.5,0.5,0.5, 0.5,0.5,0.5]).to(device)
                    elif session_label[b_idx] == 1:
                        w = torch.Tensor([1,1,1, 0.5,0.5,0.5, 0.5,0.5,0.5]).to(device)
                    elif session_label[b_idx] == 2:
                        w = torch.Tensor([0.5,0.5,0.5, 1,1,1, 0.5,0.5,0.5]).to(device)
                    elif session_label[b_idx] == 3:
                        w = torch.Tensor([0.5,0.5,0.5, 0.5,0.5,0.5, 1,1,1]).to(device)
                    buy_loss[b_idx] = buy_loss[b_idx] * w
                buy_loss = torch.mean(buy_loss)
                click_loss = click_criterion(click_outputs, click_label.float())

                valid_running_sess_loss += sess_loss.item()
                valid_running_buy_loss += buy_loss.item()
                valid_running_click_loss += click_loss.item()

                _, sess_predicted = torch.max(sess_outputs.data, 1)
                sess_acc = (sess_predicted == session_label).sum().item() / valid_batch_size
                buy_acc1, buy_acc2, buy_acc_rule2, buy_real_acc1, buy_real_acc2, buy_real_acc_rule2 = binary_acc(sess_predicted, buy_outputs, label)

                valid_click_acc += click_acc(click_outputs, click_label)

                valid_sess_acc_sum += sess_acc
                valid_buy_acc1_sum += buy_acc1
                valid_buy_real_acc1_sum += buy_real_acc1
                valid_buy_acc2_sum += buy_acc2
                valid_buy_real_acc2_sum += buy_real_acc2
                valid_buy_acc_rule2_sum += buy_acc_rule2
                valid_buy_real_acc_rule2_sum += buy_real_acc_rule2
                valid_cnt += 1

                # valid_cnt_session_0 += torch.sum(session_label == 0)
                # valid_cnt_session_1 += torch.sum(session_label == 1)
                # valid_cnt_session_2 += torch.sum(session_label == 2)
                # valid_cnt_session_3 += torch.sum(session_label == 3)

            valid_acc = valid_buy_real_acc2_sum / valid_cnt
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                valid_acc = round(valid_acc, 6)
                with open(f'{tb_path}/val_best_acc.txt', 'w') as fp:
                    print('epoch:', epoch_idx, file=fp)
                    print('batches_done:', batches_done, file=fp)
                    print('buy real acc2:', valid_acc, file=fp)
                torch.save(model, f'{tb_path}/val_best.pth')

            print('----- VAL -----')
            print('- sess loss:', valid_running_sess_loss / valid_cnt)
            print('- buy loss:',  valid_running_buy_loss / valid_cnt)
            print('- click loss:',  valid_running_click_loss / valid_cnt)
            print('- sess acc:',      valid_sess_acc_sum / valid_cnt)
            print('- buy acc1:',      valid_buy_acc1_sum.cpu().item() / valid_cnt)
            print('- buy real acc1:', valid_buy_real_acc1_sum / valid_cnt)
            print('- buy acc2:',      valid_buy_acc2_sum.cpu().item() / valid_cnt)
            print('- buy real acc2:', valid_buy_real_acc2_sum / valid_cnt)
            print('- buy acc rule2:', valid_buy_acc_rule2_sum.cpu().item() / valid_cnt)
            print('- buy real acc rule2:', valid_buy_real_acc_rule2_sum / valid_cnt)
            print('- click acc:',      valid_click_acc / valid_cnt)
            # print('valid sess cnt:', valid_cnt_session_0, valid_cnt_session_1, valid_cnt_session_2, valid_cnt_session_3)

            tb_writer.add_scalar('val/sess loss', valid_running_sess_loss / valid_cnt, batches_done)
            tb_writer.add_scalar('val/buy loss',  valid_running_buy_loss / valid_cnt, batches_done)
            tb_writer.add_scalar('val/click loss',  valid_running_click_loss / valid_cnt, batches_done)
            tb_writer.add_scalar('val/sess acc',  valid_sess_acc_sum / valid_cnt, batches_done)
            tb_writer.add_scalar('val/buy acc1',  valid_buy_acc1_sum.cpu().item() / valid_cnt, batches_done)
            tb_writer.add_scalar('val/buy real acc1', valid_buy_real_acc1_sum / valid_cnt, batches_done)
            tb_writer.add_scalar('val/buy acc2',      valid_buy_acc2_sum.cpu().item() / valid_cnt, batches_done)
            tb_writer.add_scalar('val/buy real acc2', valid_buy_real_acc2_sum / valid_cnt, batches_done)
            tb_writer.add_scalar('val/buy acc rule2', valid_buy_acc_rule2_sum.cpu().item() / valid_cnt, batches_done)
            tb_writer.add_scalar('val/buy real acc rule2', valid_buy_real_acc_rule2_sum / valid_cnt, batches_done)
            tb_writer.add_scalar('val/click acc',  valid_click_acc / valid_cnt, batches_done)

    valid_acc = round(valid_acc, 6)
    torch.save(model, f'{tb_path}/model_epoch{epoch_idx+1}_{valid_acc}.pth')

print('Finished Training')
```

```python id="_jJnvuiRs7m4"

```

<!-- #region id="fs0871p6l08l" -->
## test
<!-- #endregion -->

```python id="z2e8vb-bC0eE"
def load_test_data(data_path='/content/',
                   filename='track1_testset.csv'):
    test_samples = []
    df_test = pd.read_parquet(f'{data_path}/{filename}', sep=' ')

    total_num = int(df_test.shape[0])

    for i in tqdm(range(total_num)):
        if df_test.at[i, 'user_click_history'] == '0:0':
            user_click_list = [0]
        else:
            user_click_list = df_test.at[i, 'user_click_history'].split(',')
            user_click_list = [int(sample.split(':')[0]) for sample in user_click_list]
        num_user_click_history = len(user_click_list)
        tmp = np.zeros(400, dtype=np.int64)
        tmp[:len(user_click_list)] = user_click_list
        user_click_list = tmp
        
        exposed_items = [int(s) for s in df_test.at[i, 'exposed_items'].split(',')]

        user_portrait = [int(s) for s in df_test.at[i, 'user_protrait'].split(',')]
        # portraitidx_to_idx_dict_list: list of 10 dict, int:int
        for j in range(10):
            user_portrait[j] = portraitidx_to_idx_dict_list[j][user_portrait[j]]
        one_sample = {
            'user_click_list': user_click_list,
            'num_user_click_history': num_user_click_history,
            'user_portrait': np.array(user_portrait, dtype=np.int64),
            'item_id': np.array(exposed_items, dtype=np.int64),
            't': df_train.at[i, 'time'] # int
        }
        test_samples.append(one_sample)
    return test_samples



class BigDataCupTestDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 item_features,
                 database
                ):
        super().__init__()
        self.item_features = item_features
        self.database = database

    def __len__(self, ):
        return len(self.database)

    def __getitem__(self, idx):
        one_sample = self.database[idx]
        user_click_history = one_sample['user_click_list']
        num_user_click_history = one_sample['num_user_click_history']
        user_discrete_feature = one_sample['user_portrait']
        item_id = one_sample['item_id']
        t = one_sample['t'] # int

        a = datetime.fromtimestamp(t, tz_bj)
        month   = int(a.strftime('%m')) - 1 # 几月 01, 02, …, 12 (-1)
        date    = int(a.strftime('%d')) - 1 # 几号 01, 02, …, 31 (-1)
        weekday = int(a.strftime('%w')) # 星期几 0, 1, …, 6. 0:sunday, 6:saturday
        hour    = int(a.strftime('%H')) # 小时 00, 01, …, 23

        user_click_history_discrete_feature = np.zeros((400, 3+1 + 2+1)).astype(np.int64)
        for i in range(num_user_click_history):
            if user_click_history[i] == 0:
                user_click_history_discrete_feature[i] = self.item_features[user_click_history[i]]
            else:
                user_click_history_discrete_feature[i] = self.item_features[user_click_history[i]]

        item_discrete_feature = np.zeros((9, 3+1 + 2+1)).astype(np.int64)
        for i in range(9):
            item_discrete_feature[i] = self.item_features[item_id[i]]

        user_click_history = torch.IntTensor(user_click_history)
        user_click_history_discrete_feature = torch.IntTensor(user_click_history_discrete_feature)
        num_user_click_history = torch.IntTensor([num_user_click_history])
        user_discrete_feature = torch.IntTensor(user_discrete_feature)
        item_id = torch.IntTensor(item_id)
        item_discrete_feature = torch.IntTensor(item_discrete_feature)

        return user_click_history, \
            user_click_history_discrete_feature, \
            num_user_click_history, \
            item_id, item_discrete_feature, \
            user_discrete_feature, \
            month, date, weekday, hour

```

```python colab={"base_uri": "https://localhost:8080/"} id="xBTlmys4XqKF" executionInfo={"status": "ok", "timestamp": 1629797536637, "user_tz": -480, "elapsed": 18637, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="992eb658-61f4-4015-8f64-39144361993e"
test_samples = load_test_data(data_path='/content/', 
                                              filename='track1_testset.csv')

test_ds = BigDataCupTestDataset(item_features, test_samples)
test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=32, shuffle=False)
```

```python id="aBGsTXabBX36"
# tb_path = 'runs/2021-08-16-07:50:46-4sess_pred_item_feat_extracion_deepermodel_augorder_adamlr0.001_epoch10_lr0.0001_epoch20'
# tb_path = 'runs/2021-08-18-17:25:49-4sess_pred_item_feat_extracion_deepermodel_augorder_itemalldiscretefeat_adamlr0.001_epoch30_lr0.0001_epoch50'
# tb_path = 'runs/2021-08-19-14:52:47-transformer_augorder_itemalldiscretefeat_adamlr0.001_epoch30'
# tb_path = 'runs/2021-08-20-11:39:18-multitask_transformer_augorder_adamlr0.001_epoch10_lr0.0001_epoch20'
# tb_path = 'runs/2021-08-21-15:51:42-multitask_transformer_augorder_reweight_adamlr0.001_epoch10_lr0.0001_epoch20'
tb_path = 'runs/2021-08-22-08:01:17-multitask_transformer_augorder_reweight_timefeat_adamlr0.001_epoch10_lr0.0001_epoch15'

# model = torch.load(f'{tb_path}/model_0.0001.pth', map_location='cpu')
# model = torch.load(f'{tb_path}/model.pth', map_location='cpu')
# model = torch.load(f'{tb_path}/val_best.pth', map_location='cpu')
# model = torch.load(f'{tb_path}/val_best_epoch16_0.396906.pth', map_location='cpu')
# model = torch.load(f'{tb_path}/val_best_epoch11_0.395099.pth', map_location='cpu')
# model = torch.load(f'{tb_path}/val_best.pth', map_location='cpu')
model = torch.load(f'{tb_path}/model_epoch12_0.396327.pth', map_location='cpu')

# tta augorder
tta_augorder = False


# fp = open(f'{tb_path}/output_test_tta_augorder3_val_best.csv', 'w')
# fp = open(f'{tb_path}/output_test_epoch16_0.396906.csv', 'w')
fp = open(f'{tb_path}/output_test_epoch12.csv', 'w')
print('id,category', file=fp)


model = model.eval()
model = model.to('cpu')
model.device = 'cpu'
```

```python id="qQryhtqFCV3r"

# aug items within sess
from itertools import permutations
from functools import reduce
import operator
import random

perm1 = list(permutations([0, 1, 2]))
perm2 = list(permutations([3, 4, 5]))
perm3 = list(permutations([6, 7, 8]))

aug_order = []
for p1 in perm1:
    # print(p1)
    for p2 in perm2:
        # print(p1, p2)
        for p3 in perm3:
            # print(p1, p2, p3)
            tmp = reduce(operator.concat, [p1, p2, p3])
            aug_order.append(tmp)
len_aug_order = len(aug_order)
```

```python colab={"base_uri": "https://localhost:8080/"} id="fbDGC0xUl7RX" executionInfo={"status": "ok", "timestamp": 1629798127294, "user_tz": -480, "elapsed": 524106, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="1cf51afa-fffe-4494-aacd-deb26dd58563"
bs = 32

for i, data in tqdm(enumerate(test_dl, 0)):
    user_click_history, \
        user_click_history_discrete_feature, \
        num_user_click_history, \
        item_id, item_discrete_feature, \
        user_discrete_feature, \
        month, date, weekday, hour = data
    

    if not tta_augorder:
        sess_outputs, buy_outputs = model(
            user_click_history,
            user_click_history_discrete_feature,
            num_user_click_history,
            item_id,
            item_discrete_feature,
            user_discrete_feature,
            month, date, weekday, hour
        )

        y_pred_tag = torch.round(torch.sigmoid(buy_outputs))
        _, sess_pred = torch.max(sess_outputs.data, 1)


    else:
        sum_sess_outputs = None
        sum_buy_outputs = None

        total_aug_num = 3
        aug_order_shuffle = shuffle(aug_order)
        aug_order_shuffle = aug_order_shuffle[:total_aug_num]
        for aug_idx, ao in enumerate(aug_order_shuffle):
            ao = list(ao)
            ao_inv = np.argsort(ao)
            sess_outputs, buy_outputs = model(
                user_click_history,
                user_click_history_discrete_feature,
                num_user_click_history,
                item_id[:, ao],
                item_discrete_feature[:, ao, :],
                user_discrete_feature,
                month, date, weekday, hour
            )
            buy_outputs = buy_outputs[:, ao_inv]
            if aug_idx == 0:
                sum_sess_outputs = nn.functional.softmax(sess_outputs, dim=1)
                sum_buy_outputs = torch.sigmoid(buy_outputs)
            else:
                sum_sess_outputs += nn.functional.softmax(sess_outputs, dim=1)
                sum_buy_outputs += torch.sigmoid(buy_outputs)
        sess_outputs = sum_sess_outputs / total_aug_num
        buy_outputs = sum_buy_outputs / total_aug_num

        y_pred_tag = torch.round(buy_outputs)
        _, sess_pred = torch.max(sess_outputs.data, 1)

    for j in range(y_pred_tag.shape[0]):
        if sess_pred[j] == 0:
            y_pred_tag[j][:] = 0
        elif sess_pred[j] == 1:
            y_pred_tag[j][3:] = 0
        elif sess_pred[j] == 2:
            y_pred_tag[j][:3] = 1
            y_pred_tag[j][6:] = 0
        elif sess_pred[j] == 3:
            y_pred_tag[j][:6] = 1

        tmp = list(y_pred_tag[j].detach().numpy().astype(np.int32))
        tmp = [str(a) for a in tmp]
        p = ' '.join(tmp)
        print(f'{i * bs + j + 1},{p}', file=fp)
    # break

fp.close()
```

```python id="uY-NWb910UJH" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629502371631, "user_tz": -480, "elapsed": 763, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="a149aa76-f5d0-484a-c2ba-c91e97ad5c3b"
tta_augorder
```

```python colab={"base_uri": "https://localhost:8080/"} id="pXNsdPJj8RaY" executionInfo={"status": "ok", "timestamp": 1629553245352, "user_tz": -480, "elapsed": 1053, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="01696dc8-3f8a-459b-9bfc-47bcfd00808a"
!tail /content/drive/MyDrive/202108-bigdatacup2021/runs/2021-08-20-11:39:18-multitask_transformer_augorder_adamlr0.001_epoch10/output_test_tta_augorder3_val_best.csv
```

<!-- #region id="48tdWQuI0UqW" -->
## validation analysis
<!-- #endregion -->

```python id="VAXYAVTKoEWp"
m = torch.load('4sess_pred_item_feat_extracion_deepermodel_epoch2.pth', map_location='cpu')
m.device = 'cpu'
```

```python id="XZyAyicFzfNH"
m = model.eval()
m = model.to('cpu')
m.device = 'cpu'
```

```python colab={"base_uri": "https://localhost:8080/"} id="lyGOaITH2aZe" executionInfo={"status": "ok", "timestamp": 1629114549911, "user_tz": -480, "elapsed": 20731, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="b99ceb40-bdbc-4fd0-dd0f-2d6017aab349"
train_samples, val_samples = load_train_data()

# train_ds = BigDataCupDataset(item_info_dict, train_samples)
# train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=32, shuffle=True)

val_ds = BigDataCupDataset(item_info_dict, val_samples, train_val='val')
val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=32, shuffle=False)
```

<!-- #region id="y9F6v8jHs_fN" -->
### tta, augorder
<!-- #endregion -->

```python id="cPcKJAjt1b9Z"
a = np.array([1,2,3,4,5,6,7,8,9])
```

```python colab={"base_uri": "https://localhost:8080/"} id="w9jgxazS2RyS" executionInfo={"status": "ok", "timestamp": 1629115470397, "user_tz": -480, "elapsed": 804, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="ff5c6322-66d4-43dd-b276-58c5e5f85bff"
permutation = list(aug_order[100])
permutation
```

```python colab={"base_uri": "https://localhost:8080/"} id="zxfcWcv_2Pxf" executionInfo={"status": "ok", "timestamp": 1629115471133, "user_tz": -480, "elapsed": 5, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="2f38df0d-dd60-4720-d1fb-c11e45cf099c"
np.argsort(permutation)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1Os99rsJ1eiv" executionInfo={"status": "ok", "timestamp": 1629115474572, "user_tz": -480, "elapsed": 600, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="11c3687c-4d43-48de-a9db-a19c9b51b0e0"
a[permutation]
```

```python colab={"base_uri": "https://localhost:8080/"} id="3gws2Epl1jys" executionInfo={"status": "ok", "timestamp": 1629115544451, "user_tz": -480, "elapsed": 437, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="211f14db-6d71-4e4b-c529-a2e445ceb8b7"
a[permutation][np.argsort(permutation)]
```

```python id="SwF_cCTD42s4"
def binary_acc_nosigmoid(sess_pred, y_pred, y_test):
    # print(sess_pred)
    y_pred_tag = torch.round(y_pred)
    y_pred_tag_intact = y_pred_tag.clone()


    ##################################
    ## vanilla
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc1 = correct_results_sum / y_test.shape[0] / 9

    real_acc1 = 0.0
    for i in range(y_test.shape[0]):
        correct_results_sum = (y_pred_tag[i] == y_test[i]).sum().float()
        one_acc = correct_results_sum / 9
        if one_acc == 1:
            real_acc1 += 1
    real_acc1 = real_acc1 / y_test.shape[0]
    # print(y_pred_tag)

    ####################################
    ## use sess to refine y_pred_tag
    for i in range(y_test.shape[0]):
        if sess_pred[i] == 0:
            y_pred_tag[i][:] = 0
        elif sess_pred[i] == 1:
            y_pred_tag[i][3:] = 0
        elif sess_pred[i] == 2:
            y_pred_tag[i][:3] = 1
            y_pred_tag[i][6:] = 0
        elif sess_pred[i] == 3:
            y_pred_tag[i][:6] = 1

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc2 = correct_results_sum / y_test.shape[0] / 9

    real_acc2 = 0.0
    for i in range(y_test.shape[0]):
        correct_results_sum = (y_pred_tag[i] == y_test[i]).sum().float()
        one_acc = correct_results_sum / 9
        if one_acc == 1:
            real_acc2 += 1
    real_acc2 = real_acc2 / y_test.shape[0]
    # print(y_pred_tag)

    #######################################
    ## rule 2
    y_pred_tag = y_pred_tag_intact
    acc_rule2 = 0.0
    real_acc_rule2 = 0.0
    for i in range(y_test.shape[0]):
        for j in range(9):
            k = 8 - j
            if k >= 6 and y_pred_tag[i][k] == 1:
                y_pred_tag[i][:6] = 1
            if k >= 3 and y_pred_tag[i][k] == 1:
                y_pred_tag[i][:3] = 1
        
        correct_results_sum = (y_pred_tag[i] == y_test[i]).sum().float()
        a = correct_results_sum / 9
        acc_rule2 += a
        if a == 1:
            real_acc_rule2 += 1
    acc_rule2 = acc_rule2 / y_test.shape[0]
    real_acc_rule2 = real_acc_rule2 / y_test.shape[0]
    # print(y_pred_tag)


    return acc1, acc2, acc_rule2, real_acc1, real_acc2, real_acc_rule2
```

```python colab={"base_uri": "https://localhost:8080/"} id="tvIsmHR2tDvU" executionInfo={"status": "ok", "timestamp": 1629118299943, "user_tz": -480, "elapsed": 144123, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="dbce97cf-2467-4280-fb46-9bcc74ac448c"


# aug items within sess
from itertools import permutations
from functools import reduce
import operator
import random

perm1 = list(permutations([0, 1, 2]))
perm2 = list(permutations([3, 4, 5]))
perm3 = list(permutations([6, 7, 8]))

aug_order = []
for p1 in perm1:
    # print(p1)
    for p2 in perm2:
        # print(p1, p2)
        for p3 in perm3:
            # print(p1, p2, p3)
            tmp = reduce(operator.concat, [p1, p2, p3])
            aug_order.append(tmp)
len_aug_order = len(aug_order)


for trial in range(10):

    sess_pred_list = []
    sess_gt_list = []

    one_zero  = np.zeros(9) # 本来买了1，预测称没买0
    zero_one  = np.zeros(9) # 本来没买0，预测成购买1
    one_one   = np.zeros(9) # 本来买了，预测成买了
    zero_zero = np.zeros(9) # 本来没买，预测称没买

    pred_num_list = []
    gt_num_list = []

    valid_cnt = 0
    valid_sess_acc_sum = 0
    valid_buy_acc1_sum = 0
    valid_buy_real_acc1_sum = 0
    valid_buy_acc2_sum = 0
    valid_buy_real_acc2_sum = 0
    valid_buy_acc_rule2_sum = 0
    valid_buy_real_acc_rule2_sum = 0

    valid_buy_acc1_gtsess_sum = 0
    valid_buy_real_acc_gtsess_sum = 0


    for i, data in tqdm(enumerate(val_dl, 0)):
        user_click_history, \
            user_click_history_discrete_feature, user_click_history_cont_feature, \
            num_user_click_history, \
            item_id, item_discrete_feature, item_cont_feature, \
            user_discrete_feature, label, session_label = data

        sum_sess_outputs = None
        sum_buy_outputs = None

        total_aug_num = 2
        aug_order_shuffle = shuffle(aug_order)
        aug_order_shuffle = aug_order_shuffle[:total_aug_num]
        for aug_idx, ao in enumerate(aug_order_shuffle):
            ao = list(ao)
            ao_inv = np.argsort(ao)

            sess_outputs, buy_outputs = model(
                user_click_history,
                user_click_history_discrete_feature,
                user_click_history_cont_feature,
                num_user_click_history,
                item_id[:, ao],
                item_discrete_feature[:, ao, :],
                item_cont_feature[:, ao, :],
                user_discrete_feature
            )
            buy_outputs = buy_outputs[:, ao_inv]

            if aug_idx == 0:
                sum_sess_outputs = nn.functional.softmax(sess_outputs, dim=1)
                sum_buy_outputs = torch.sigmoid(buy_outputs)
            else:
                sum_sess_outputs += nn.functional.softmax(sess_outputs, dim=1)
                sum_buy_outputs += torch.sigmoid(buy_outputs)

        sess_outputs = sum_sess_outputs / total_aug_num
        buy_outputs = sum_buy_outputs / total_aug_num


        bs = user_click_history.shape[0]

        ## let all 0,1,2 item buy (this will reduce performance, tested)
        # buy_outputs[:, :3] = 1 

        _, sess_predicted = torch.max(sess_outputs.data, 1)
        sess_acc = (sess_predicted == session_label).sum().item() / bs
        buy_acc1, buy_acc2, buy_acc_rule2, buy_real_acc1, buy_real_acc2, buy_real_acc_rule2 = binary_acc_nosigmoid(sess_predicted, buy_outputs, label)
        _, buy_acc1_gtsess, _, _, buy_real_acc_gtsess, _ = binary_acc_nosigmoid(session_label, buy_outputs, label)

        sess_pred_list.extend(list(sess_predicted.numpy()))
        sess_gt_list.extend(list(session_label))

        y_pred_tag = torch.round(buy_outputs).detach().numpy() # note rm sigmoid here, 
        label = label.numpy()

        pred_num = np.sum(y_pred_tag, axis=1)
        gt_num = np.sum(label, axis=1)
        pred_num_list.extend(list(pred_num))
        gt_num_list.extend(list(gt_num))

        valid_sess_acc_sum += sess_acc
        valid_buy_acc1_sum += buy_acc1
        valid_buy_real_acc1_sum += buy_real_acc1
        valid_buy_acc2_sum += buy_acc2
        valid_buy_real_acc2_sum += buy_real_acc2
        valid_buy_acc_rule2_sum += buy_acc_rule2
        valid_buy_real_acc_rule2_sum += buy_real_acc_rule2

        valid_buy_acc1_gtsess_sum += buy_acc1_gtsess
        valid_buy_real_acc_gtsess_sum += buy_real_acc_gtsess
        valid_cnt += 1

        for b in range(bs):
            y_pred = y_pred_tag[b]
            y_gt = label[b]
            for i in range(9):
                if y_pred[i] == 1 and y_gt[i] == 1:
                    one_one[i] += 1
                elif y_pred[i] == 0 and y_gt[i] == 0:
                    zero_zero[i] += 1
                elif y_pred[i] == 1 and y_gt[i] == 0:
                    one_zero[i] += 1
                elif y_pred[i] == 0 and y_gt[i] == 1:
                    zero_one[i] += 1


    print('----- VAL -----')
    print('- sess acc:', valid_sess_acc_sum / valid_cnt)
    print('- buy acc1:', valid_buy_acc1_sum / valid_cnt)
    print('- buy real acc1:', valid_buy_real_acc1_sum / valid_cnt)
    print('- buy acc2:', valid_buy_acc2_sum / valid_cnt)
    print('- buy real acc2:', valid_buy_real_acc2_sum / valid_cnt)
    print('- buy acc rule2:', valid_buy_acc_rule2_sum / valid_cnt)
    print('- buy real acc rule2:', valid_buy_real_acc_rule2_sum / valid_cnt)

    print('- buy acc1 gtsess:', valid_buy_acc1_gtsess_sum / valid_cnt)
    print('- buy real acc gtsess:', valid_buy_real_acc_gtsess_sum / valid_cnt)

```

<!-- #region id="n9qpwUxzr4Bn" -->
### result analysis
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="s9rLPFfPzhmw" executionInfo={"status": "ok", "timestamp": 1629115982236, "user_tz": -480, "elapsed": 11834, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="c78c695e-f027-4f59-a361-189e755fd1f3"
model = m.eval()
sess_pred_list = []
sess_gt_list = []

one_zero  = np.zeros(9) # 本来买了1，预测称没买0
zero_one  = np.zeros(9) # 本来没买0，预测成购买1
one_one   = np.zeros(9) # 本来买了，预测成买了
zero_zero = np.zeros(9) # 本来没买，预测称没买

pred_num_list = []
gt_num_list = []

valid_cnt = 0
valid_sess_acc_sum = 0
valid_buy_acc1_sum = 0
valid_buy_real_acc1_sum = 0
valid_buy_acc2_sum = 0
valid_buy_real_acc2_sum = 0
valid_buy_acc_rule2_sum = 0
valid_buy_real_acc_rule2_sum = 0

valid_buy_acc1_gtsess_sum = 0
valid_buy_real_acc_gtsess_sum = 0

for i, data in tqdm(enumerate(val_dl, 0)):
    user_click_history, \
        user_click_history_discrete_feature, user_click_history_cont_feature, \
        num_user_click_history, \
        item_id, item_discrete_feature, item_cont_feature, \
        user_discrete_feature, label, session_label = data
    sess_outputs, buy_outputs = model(
        user_click_history,
        user_click_history_discrete_feature,
        user_click_history_cont_feature,
        num_user_click_history,
        item_id,
        item_discrete_feature,
        item_cont_feature,
        user_discrete_feature
    )
    bs = user_click_history.shape[0]

    ## let all 0,1,2 item buy (this will reduce performance, tested)
    # buy_outputs[:, :3] = 1 

    _, sess_predicted = torch.max(sess_outputs.data, 1)
    sess_acc = (sess_predicted == session_label).sum().item() / bs
    buy_acc1, buy_acc2, buy_acc_rule2, buy_real_acc1, buy_real_acc2, buy_real_acc_rule2 = binary_acc(sess_predicted, buy_outputs, label)
    _, buy_acc1_gtsess, _, _, buy_real_acc_gtsess, _ = binary_acc(session_label, buy_outputs, label)

    y_pred_tag = torch.round(torch.sigmoid(buy_outputs)).detach().numpy()
    label = label.numpy()

    pred_num = np.sum(y_pred_tag, axis=1)
    gt_num = np.sum(label, axis=1)
    pred_num_list.extend(list(pred_num))
    gt_num_list.extend(list(gt_num))

    valid_sess_acc_sum += sess_acc
    valid_buy_acc1_sum += buy_acc1
    valid_buy_real_acc1_sum += buy_real_acc1
    valid_buy_acc2_sum += buy_acc2
    valid_buy_real_acc2_sum += buy_real_acc2
    valid_buy_acc_rule2_sum += buy_acc_rule2
    valid_buy_real_acc_rule2_sum += buy_real_acc_rule2

    valid_buy_acc1_gtsess_sum += buy_acc1_gtsess
    valid_buy_real_acc_gtsess_sum += buy_real_acc_gtsess
    valid_cnt += 1

    for b in range(bs):
        y_pred = y_pred_tag[b]
        y_gt = label[b]
        for i in range(9):
            if y_pred[i] == 1 and y_gt[i] == 1:
                one_one[i] += 1
            elif y_pred[i] == 0 and y_gt[i] == 0:
                zero_zero[i] += 1
            elif y_pred[i] == 1 and y_gt[i] == 0:
                one_zero[i] += 1
            elif y_pred[i] == 0 and y_gt[i] == 1:
                zero_one[i] += 1

    _, sess_pred = torch.max(sess_outputs.data, 1)
    sess_pred_list.extend(list(sess_pred.numpy()))
    sess_gt_list.extend(list(session_label))


print('----- VAL -----')
print('- sess acc:', valid_sess_acc_sum / valid_cnt)
print('- buy acc1:', valid_buy_acc1_sum / valid_cnt)
print('- buy real acc1:', valid_buy_real_acc1_sum / valid_cnt)
print('- buy acc2:', valid_buy_acc2_sum / valid_cnt)
print('- buy real acc2:', valid_buy_real_acc2_sum / valid_cnt)
print('- buy acc rule2:', valid_buy_acc_rule2_sum / valid_cnt)
print('- buy real acc rule2:', valid_buy_real_acc_rule2_sum / valid_cnt)

print('- buy acc1 gtsess:', valid_buy_acc1_gtsess_sum / valid_cnt)
print('- buy real acc gtsess:', valid_buy_real_acc_gtsess_sum / valid_cnt)

```

```python id="jPtKmxUT_0y2"

```

```python id="1B7xY2Oc2_ee"
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

```

```python colab={"base_uri": "https://localhost:8080/", "height": 541} id="rSZVhBGJ3hFR" executionInfo={"status": "ok", "timestamp": 1629112874016, "user_tz": -480, "elapsed": 542, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="d62c079a-c35d-426f-fb2b-6d34a19563e9"
# Confusion matrix whose i-th row and j-th column entry indicates 
# the number of samples with 
# true label being i-th class, and 
# predicted label being j-th class.
a = confusion_matrix(sess_gt_list, sess_pred_list)
a_per = a / np.sum(a, axis=1, keepdims=True) * 100
cm_display = ConfusionMatrixDisplay(a, display_labels=range(4)).plot(values_format='d')
cm_display = ConfusionMatrixDisplay(a_per, display_labels=range(4)).plot(values_format='2.0f')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 541} id="WLY2gryVB1aH" executionInfo={"status": "ok", "timestamp": 1629112898996, "user_tz": -480, "elapsed": 2829, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="32ad0166-a22e-4d0c-f49b-d927a6449c93"
a = confusion_matrix(pred_num_list, gt_num_list)
a_per = a / np.sum(a, axis=1, keepdims=True) * 100
cm_display = ConfusionMatrixDisplay(a, display_labels=range(10)).plot(values_format='d')
cm_display = ConfusionMatrixDisplay(a_per, display_labels=range(10)).plot(values_format='2.0f')
```

```python colab={"base_uri": "https://localhost:8080/"} id="C-kfakmqB3kP" executionInfo={"status": "ok", "timestamp": 1629034673004, "user_tz": -480, "elapsed": 9, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="006de0bf-40df-40c4-9a37-d2df52367d45"
s = 0
for i in range(10):
    s += a[i][i]
print(s)
```

```python colab={"base_uri": "https://localhost:8080/"} id="WDCD7YFJCCji" executionInfo={"status": "ok", "timestamp": 1629034706330, "user_tz": -480, "elapsed": 284, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="891b4968-18c1-466d-b5d6-5e676745bc13"
4605 / np.sum(a)
```

```python colab={"base_uri": "https://localhost:8080/"} id="NHtQR8-ICQBJ" executionInfo={"status": "ok", "timestamp": 1629034710748, "user_tz": -480, "elapsed": 265, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="3b86497a-25a9-4a37-8923-c39a659ee2b6"
np.sum(a)
```

```python colab={"base_uri": "https://localhost:8080/"} id="VPt69Xkk4XgA" executionInfo={"status": "ok", "timestamp": 1629034414975, "user_tz": -480, "elapsed": 254, "user": {"displayName": "Tzu-Heng Lin", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiD2y1K7AjV8IekJ4j57pQP6AoOEiHbcW-XWKzorw=s64", "userId": "11900644505227993241"}} outputId="2c4aaac0-2f28-498e-f9dd-c8604edbb986"
a = one_zero + zero_one + one_one + zero_zero
print(one_zero)
print(zero_one)
print(one_one)
print(zero_zero) 
print('')
print(np.round(one_zero  / a, 2))
print(np.round(zero_one  / a, 2))
print(np.round(one_one   / a, 2))
print(np.round(zero_zero / a, 2)) 
```

```python id="An-TH5M6AhKL"

```
