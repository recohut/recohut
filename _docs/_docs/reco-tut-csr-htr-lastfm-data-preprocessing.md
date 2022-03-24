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

```python id="d1o6gvXino6X" executionInfo={"status": "ok", "timestamp": 1629801652998, "user_tz": -330, "elapsed": 950, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-csr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="8MpUlXlWny29" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629801655924, "user_tz": -330, "elapsed": 2937, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="949ca5b7-8ff1-452a-936f-f4d300ba55c5"
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

```python id="ochl7m4B-XB3"
!git pull --rebase origin "{branch}"
```

```python id="jrUWb0jiny3G"
!git status
```

```python id="MpYWa13ony3I" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629804411250, "user_tz": -330, "elapsed": 1997, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fd09ac32-7a59-47c6-9cae-aaf74975fbd3"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="XlHj9DTvojCE"
# !pip install -q dvc dvc[gdrive]
!dvc pull
```

```python id="yyixkXS6yuHZ"
!dvc commit && dvc push
```

<!-- #region id="EVK1iRiDlLai" -->
---
<!-- #endregion -->

```python id="XmZkWNGpwppR" executionInfo={"status": "ok", "timestamp": 1629802118482, "user_tz": -330, "elapsed": 668, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
import copy
import pickle
from scipy.sparse import coo_matrix
import scipy.sparse
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="qYkIMHIRlUUk" executionInfo={"status": "ok", "timestamp": 1629804339670, "user_tz": -330, "elapsed": 594, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="52ccd769-b811-4275-a222-8b15b548bd3d"
df = pd.read_csv('./data/bronze/lastfm/user_artists.dat', delimiter="\t", header=0, dtype=np.int32)
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="e807uoHVwv6a" executionInfo={"status": "ok", "timestamp": 1629804340954, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fe7dba9b-ec4d-4860-b6fc-221ed330075f"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="C_1YoJJpwwbb" executionInfo={"status": "ok", "timestamp": 1629804344987, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4a9237a2-2d31-47d3-99ff-e5596fa93e21"
df.drop(['weight'], axis=1, inplace=True)
user_list = np.unique(df['userID'])
item_list = np.unique(df['artistID'])
df.rename(columns={'userID': 'uid', 'artistID': 'iid'}, inplace=True)
df.head()
```

```python id="qbfc0fDcwxco" executionInfo={"status": "ok", "timestamp": 1629804344988, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
np.random.shuffle(user_list)
userId_old2new_list = np.zeros(np.max(user_list) + 1)
userId_new2old_list = np.zeros_like(user_list)
i = 0
for u in user_list:
    userId_old2new_list[u] = i
    userId_new2old_list[i] = u
    i += 1

np.random.shuffle(item_list)
itemId_old2new_list = np.zeros(np.max(item_list) + 1)
itemId_new2old_list = np.zeros_like(item_list)
j = 0
for i in item_list:
    itemId_old2new_list[i] = j
    itemId_new2old_list[j] = i
    j += 1

u_array = df['uid'].values
i_array = df['iid'].values
u_array_new = userId_old2new_list[u_array]
i_array_new = itemId_old2new_list[i_array]
df['uid'] = u_array_new
df['iid'] = i_array_new
```

<!-- #region id="Tylz6wWl87F_" -->
For LastFM, we randomly select 10% (25% of 40%) of users and all their records as the validation set and 30% (75% of 40%) of items and all their records as the test set.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0svr0QY9xE9B" executionInfo={"status": "ok", "timestamp": 1629804345591, "user_tz": -330, "elapsed": 618, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6087a58f-03e2-476d-8f59-c00dd10aed53"
user_list = np.unique(df['uid'].values)
cold_user = np.random.choice(user_list, int(len(user_list) * 0.4), replace=False)
warm_user = np.array(list(set(user_list) - set(cold_user)))

test_df = copy.copy(df)
test_df = test_df[test_df['uid'].isin(cold_user)]
train = df[df['uid'].isin(warm_user)]
vali_user = np.random.choice(cold_user, int(len(cold_user) * 0.25), replace=False)
test_user = np.array(list(set(cold_user) - set(vali_user)))
vali_df = copy.copy(test_df)
vali_df = vali_df[vali_df['uid'].isin(vali_user)]
test_df = test_df[test_df['uid'].isin(test_user)]

train.reset_index(drop=True, inplace=True)
vali_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

print('total number of user = ' + str(len(user_list)))
print('total number of item = ' + str(len(item_list)))
print('#' * 20)
print('train')
print('number of user = ' + str(len(train['uid'].unique())))
print('number of item = ' + str(len(train['iid'].unique())))
print('number of interaction = ' + str(len(train)))
print('#' * 20)
print('vali')
print('number of user = ' + str(len(vali_df['uid'].unique())))
print('number of item = ' + str(len(vali_df['iid'].unique())))
print('number of interaction = ' + str(len(vali_df)))
print('#' * 20)
print('test')
print('number of user = ' + str(len(test_df['uid'].unique())))
print('number of item = ' + str(len(test_df['iid'].unique())))
print('number of interaction = ' + str(len(test_df)))
```

```python id="i-s9p4AZxE6J" executionInfo={"status": "ok", "timestamp": 1629804355826, "user_tz": -330, "elapsed": 837, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
with open('./data/silver/lastfm/info.pkl', 'wb') as f:
    pickle.dump({'num_user': len(user_list), 'num_item': len(item_list)}, f)

train.to_csv('./data/silver/lastfm/train.csv', index=False)
vali_df.to_csv('./data/silver/lastfm/vali.csv', index=False)
test_df.to_csv('./data/silver/lastfm/test.csv', index=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="nllL_RgyyAUl" executionInfo={"status": "ok", "timestamp": 1629804355829, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f563774e-ba60-4eb6-fe68-1c29a6d48935"
friend_df = pd.read_csv('./data/bronze/lastfm/user_friends.dat', delimiter="\t", header=0, dtype=np.int32)
friend_df.head()
```

```python id="MJkMLFd6xE3C" executionInfo={"status": "ok", "timestamp": 1629804356633, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
user_array = friend_df['userID'].values
friend_array = friend_df['friendID'].values
user_array_new = userId_old2new_list[user_array]
friend_array_new = userId_old2new_list[friend_array]

friend_df['userID'] = user_array_new
friend_df['friendID'] = friend_array_new

row = friend_df['userID'].values
col = friend_df['friendID'].values
coo = coo_matrix((np.ones_like(row), (row, col)), 
                 shape=(len(user_list), len(user_list)))
scipy.sparse.save_npz('./data/silver/lastfm/user_content.npz', coo)
```

```python id="OvbkW7qk6N7B"

```
