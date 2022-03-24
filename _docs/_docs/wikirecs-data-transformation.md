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

```python id="Jep4VQyz3ZzE"
project_name="reco-wikirecs"; branch="master"; account="sparsh-ai"
```

```python id="ailHP5gi3ZzP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625769898023, "user_tz": -330, "elapsed": 6373, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2f2a562b-cd52-426d-f6e7-b8db0592c49a"
!cp /content/drive/MyDrive/mykeys.py /content
import mykeys
!rm /content/mykeys.py
path = "/content/" + project_name; 
!mkdir "{path}"
%cd "{path}"
import sys; sys.path.append(path)
!git config --global user.email "sparsh@recohut.com"
!git config --global user.name  "colab-sparsh"
!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
```

```python id="WWDDXhuK9klF"
%cd /content/reco-wikirecs/
```

```python id="6HVnZkVW3ZzQ"
!git status
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="LLMOakVK7lZg"
!pip install -r requirements.txt
```

<!-- #region id="wlWx6OrY3n_A" -->
---
<!-- #endregion -->

<!-- #region id="kiO7Fk7khazs" -->
## Setup
<!-- #endregion -->

```python id="ZsdBdQU4SlPy" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625770001691, "user_tz": -330, "elapsed": 13698, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="60619af3-c300-40fe-8c9f-e7bca99a6603"
!pip install -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="wXZfzk8rTYX6"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import itertools
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, coo_matrix

from recochef.datasets.wikirecs import WikiRecs

from utils import *
from wiki_pull import *
```

```python id="X1XNTud2orfP"
%matplotlib inline
%load_ext autoreload
%autoreload 2
```

<!-- #region id="ssPq8Lv5heto" -->
## Data loading
<!-- #endregion -->

```python id="8k7gkgbdSYrF"
wikidata = WikiRecs()
```

```python id="gMtXFbWLTpJT" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625770074180, "user_tz": -330, "elapsed": 37009, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0a11cd60-7888-4f05-9a59-e857705c98d4"
df = wikidata.load_interactions()
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="1xeCAd1jcgvx" executionInfo={"status": "ok", "timestamp": 1625770883423, "user_tz": -330, "elapsed": 654, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2d7289fc-a8bf-4f7a-f6c4-c26f550cfe3d"
df.head()
```

<!-- #region id="1Qu0d-0Qhg23" -->
## EDA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 435} id="KeeLRV8tfvOd" executionInfo={"status": "ok", "timestamp": 1625771142694, "user_tz": -330, "elapsed": 1774, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9b81f205-5de1-4460-a325-531eaf30eda3"
# Look at the distribution of edit counts
edit_counts = df.groupby('USERID').USERID.count().values

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
sns.distplot(edit_counts,kde=False,bins=np.arange(0,20000,200))
plt.xlabel('Number of edits by user')
plt.subplot(1,2,2)
sns.distplot(edit_counts,kde=False,bins=np.arange(0,200,1))
plt.xlim([0,200])
plt.xlabel('Number of edits by user')
num_counts = len(edit_counts)
print("Median edit counts: %d" % np.median(edit_counts))
thres = 5
over_thres = np.sum(edit_counts > thres)
print("Number over threshold %d: %d (%.f%%)" % (thres, over_thres, 100*over_thres/num_counts))
```

```python colab={"base_uri": "https://localhost:8080/"} id="Y36Q_2ZTgeMl" executionInfo={"status": "ok", "timestamp": 1625771206828, "user_tz": -330, "elapsed": 1866, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2ef045d2-d681-49c6-892b-f9e19fceb0de"
# Most edits by user
df.groupby(['USERID','USERNAME']).USERID.count().sort_values(ascending=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="DEWSXfV9g94l" executionInfo={"status": "ok", "timestamp": 1625771245998, "user_tz": -330, "elapsed": 1418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3f5c41c6-8c59-40a7-d961-2d43ef9c02ee"
# Find the elbow in number of edits
plt.plot(df.groupby(['USERID','USERNAME']).USERID.count().sort_values(ascending=False).values)
# plt.ylim([0,20000])
```

```python colab={"base_uri": "https://localhost:8080/"} id="z-Yg51cShHdT" executionInfo={"status": "ok", "timestamp": 1625771301248, "user_tz": -330, "elapsed": 16636, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b1e55e48-974b-44f2-c224-7c6ea76a7df5"
# What are the most popular pages (edited by the most users)
page_popularity = df.drop_duplicates(subset=['TITLE','USERNAME']).groupby('TITLE').count().USERNAME.sort_values()
page_popularity.iloc[-1000:].iloc[::-1]
```

```python colab={"base_uri": "https://localhost:8080/"} id="T0p3yQorpFWs" executionInfo={"status": "ok", "timestamp": 1625773347902, "user_tz": -330, "elapsed": 689, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3136f4a0-038c-4722-cf18-8cad5199fd3f"
df.sample().USERNAME
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="f5zITFxcozuU" executionInfo={"status": "ok", "timestamp": 1625773446016, "user_tz": -330, "elapsed": 932, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="92908c8e-2847-4d48-f1f9-3178a3bc5d53"
cols = ['userid', 'user', 'pageid', 'title',
       'timestamp', 'sizediff']

oneuser = get_edit_history(user="SanAnMan",
                            latest_timestamp="2021-07-08T22:02:09Z",
                            earliest_timestamp="2020-05-28T22:02:09Z")

oneuser = pd.DataFrame(oneuser).loc[:,cols]
oneuser
```

<!-- #region id="aF2RyBNxhiXv" -->
## Data cleaning
<!-- #endregion -->

<!-- #region id="Qu3Pz_tS-WMQ" -->
### Remove consecutive edits and summarize runs
<!-- #endregion -->

```python id="7hWZQkiX-WMR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625771653795, "user_tz": -330, "elapsed": 64566, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98ac4ee0-37a6-44cb-d89e-e28d209ba3ee"
%%time
def remove_consecutive_edits(df):
    c = dict(zip(df.columns, range(len(df.columns))))
    
    keyfunc = lambda x: (x[c['USERID']],x[c['ITEMID']])
    first_and_last = lambda run: [run[0][c['USERID']],
                                run[0][c['USERNAME']],
                                run[0][c['ITEMID']],
                                run[0][c['TITLE']],
                                run[-1][c['TIMESTAMP']],
                                run[0][c['TIMESTAMP']],
                                sum([abs(r[c['SIZEDIFF']]) for r in run]),
                                len(run)]
    d  = df.values.tolist()
    return pd.DataFrame([first_and_last(list(g)) for k,g in itertools.groupby(d, key=keyfunc)], 
                        columns=['USERID', 'USER', 'ITEMID', 'TITLE', 'FIRST_TIMESTAMP', 'LAST_TIMESTAMP','SUM_SIZEDIFF','CONSECUTIVE_EDITS'])
                        
clean_df = remove_consecutive_edits(df)
```

<!-- #region id="ybEnBJ3p-WMT" -->
### Remove top N most popular pages
<!-- #endregion -->

```python id="jOEyyMOs-WMU" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625771697564, "user_tz": -330, "elapsed": 18456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="372cec11-d432-46f4-bf33-d52962a4a222"
# Get the top most popular pages
TOPN = 20
popularpages = df.drop_duplicates(subset=['TITLE','ITEMID','USERID']).groupby(['TITLE','ITEMID']).count().USERNAME.sort_values()[-TOPN:]
popularpages
```

```python id="uSz0qP6g-WMV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625772016143, "user_tz": -330, "elapsed": 1917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f8dddfa2-c74d-4216-8492-6b6605b95a86"
# Remove those popular pages
before_count = len(df)
popular_pageids = popularpages.index.get_level_values(level='ITEMID').values
is_popular_page_edit = clean_df.ITEMID.isin(popular_pageids)
clean_df = clean_df.loc[~is_popular_page_edit].copy()
all_histories = None
after_count = len(clean_df)
print("%d edits (%.1f%%) were in top %d popular pages. Length after removing: %d" % (np.sum(is_popular_page_edit), 
                                                                                     100* np.sum(is_popular_page_edit)/before_count,
                                                                                     TOPN,
                                                                                     after_count)
     )
```

```python id="YLAq2a7I-WMZ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625772019859, "user_tz": -330, "elapsed": 687, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="860cc588-196c-4d76-dc7e-67257d66a41a"
print("Number of unique page ids: {}".format(len(clean_df.ITEMID.unique())))
```

<!-- #region id="R8hqTBZZ-WMa" -->
### Remove users with too many or too few edits
<!-- #endregion -->

```python id="WajFz7Bz-WMc"
MIN_EDITS = 5
MAX_EDITS = 10000
```

```python id="SnbOrhH9-WMd" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625772042100, "user_tz": -330, "elapsed": 1259, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="17467712-d2cd-4ee8-81fe-73901448d891"
# Get user edit counts
all_user_edit_counts = clean_df.groupby(['USERID','USER']).USERID.count()

# Remove users with too few edits
keep_user = all_user_edit_counts.values >= MIN_EDITS

# Remove users with too many edits
keep_user = keep_user & (all_user_edit_counts.values <= MAX_EDITS)

# Remove users with "bot" in the name
is_bot = ['bot' in username.lower() for username in all_user_edit_counts.index.get_level_values(1).values]
keep_user = keep_user & ~np.array(is_bot)
print("Keep %d users out of %d (%.1f%%)" % (np.sum(keep_user), len(all_user_edit_counts), 100*float(np.sum(keep_user))/len(all_user_edit_counts)))
```

```python id="nWX-64bD-WMf"
# Remove those users
userids_to_keep = all_user_edit_counts.index.get_level_values(0).values[keep_user]

clean_df = clean_df.loc[clean_df.USERID.isin(userids_to_keep)]

clean_df = clean_df.reset_index(drop=True)
```

```python id="WbfD8r0w-WMg" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625772079295, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0c2dc2e6-ab40-4d48-ff4d-4415f45f459a"
print("Length after removing users: {}".format(len(clean_df)))
```

<!-- #region id="RjNaQy7Y-WMi" -->
## Build lookup tables
<!-- #endregion -->

```python id="EGhN2Znk-WMj"
# Page id to title and back
lookup = clean_df.drop_duplicates(subset=['ITEMID']).loc[:,['ITEMID','TITLE']]
p2t = dict(zip(lookup.ITEMID, lookup.TITLE))
t2p = dict(zip(lookup.TITLE, lookup.ITEMID))

# User id to name and back
lookup = clean_df.drop_duplicates(subset=['USERID']).loc[:,['USERID','USER']]
u2n = dict(zip(lookup.USERID, lookup.USER))
n2u = dict(zip(lookup.USER, lookup.USERID))
```


```python id="lOfK-7xE-WMk"
# Page id and userid to index in cooccurence matrix and back
pageids = np.sort(clean_df.ITEMID.unique())
userids = np.sort(clean_df.USERID.unique())
 
p2i = {pageid:i for i, pageid in enumerate(pageids)}
u2i = {userid:i for i, userid in enumerate(userids)}


i2p = {v: k for k, v in p2i.items()}
i2u = {v: k for k, v in u2i.items()}
```


```python id="KOSXJpAr-WMl"
# User name and page title to index and back
n2i = {k:u2i[v] for k, v in n2u.items() if v in u2i}
t2i = {k:p2i[v] for k, v in t2p.items() if v in p2i}

i2n = {v: k for k, v in n2i.items()}
i2t = {v: k for k, v in t2i.items()}
```

<!-- #region id="QSrLvJ9W-WMn" -->

## Build test and training set
<!-- #endregion -->

```python id="ay5BdOLB-WMo"
# Make a test set from the most recent edit by each user
histories_test = clean_df.groupby(['USERID','USER'],as_index=False).first()
```

```python id="ab8sxNLt-WMo"
# Subtract it from the rest to make the training set
histories_train = dataframe_set_subtract(clean_df, histories_test)
histories_train.reset_index(drop=True, inplace=True)
```

```python id="ZVWFlnSK-WMp"
# Make a dev set from the second most recent edit by each user
histories_dev = histories_train.groupby(['USERID','USER'],as_index=False).first()
# Subtract it from the rest to make the final training set
histories_train = dataframe_set_subtract(histories_train, histories_dev)
histories_train.reset_index(drop=True, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="peWz8aTJmNGt" executionInfo={"status": "ok", "timestamp": 1625772590008, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="765d7fb4-2f3f-49cb-a2ad-b6675b7fa2db"
print("Length of test set: {}".format(len(histories_test)))
print("Length of dev set: {}".format(len(histories_dev)))
print("Length of training after removal of test: {}".format(len(histories_train)))
```

```python id="cYDo1XJM-WMr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625772682855, "user_tz": -330, "elapsed": 12712, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1c7d52b3-b616-4ca8-c810-4e0567a33a9d"
print("Number of pages in training set: {}".format(len(histories_train.ITEMID.unique())))
print("Number of users in training set: {}".format(len(histories_train.USERID.unique())))
print("Number of pages with > 1 user editing: {}".format(np.sum(histories_train.drop_duplicates(subset=['TITLE','USER']).groupby('TITLE').count().USER > 1)))
```

```python id="ht3O-0DL-WMx" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625773152503, "user_tz": -330, "elapsed": 4494, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="365a7edf-d948-4739-8705-5c1129ea27fe"
resurface_userids, discovery_userids = get_resurface_discovery(histories_train, histories_dev)

print("%d out of %d userids are resurfaced (%.1f%%)" % (len(resurface_userids), len(userids), 100*float(len(resurface_userids))/len(userids)))
print("%d out of %d userids are discovered (%.1f%%)" % (len(discovery_userids), len(userids), 100*float(len(discovery_userids))/len(userids)))
```

<!-- #region id="sgzzNkOxr8Z9" -->
## Build matrix for implicit collaborative filtering
<!-- #endregion -->

```python id="tKvJEuJNrrs-"
# Get the user/page edit counts
for_implicit = histories_train.groupby(["USERID","ITEMID"]).count().FIRST_TIMESTAMP.reset_index().rename(columns={'FIRST_TIMESTAMP':'edits'})
for_implicit.loc[:,'edits'] = for_implicit.edits.astype(np.int32)
```

```python id="78pLFLfesDF1"
row = np.array([p2i[p] for p in for_implicit.ITEMID.values])
col = np.array([u2i[u] for u in for_implicit.USERID.values])

implicit_matrix_coo = coo_matrix((for_implicit.edits.values, (row, col)))

implicit_matrix = csc_matrix(implicit_matrix_coo)
```

<!-- #region id="2GtDLqdKsx1d" -->
## Saving artifacts
<!-- #endregion -->

```python id="cX7sQzl_nNx3"
save_pickle((p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t), 'lookup_tables.pickle')
save_pickle((userids, pageids), 'users_and_pages.pickle')
save_pickle((resurface_userids, discovery_userids), 'resurface_discovery_users.pickle')
save_pickle(implicit_matrix,'implicit_matrix.pickle')
```
