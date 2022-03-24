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

<!-- #region id="GnvNDv2t5tDc" -->
# WikiRecs Part 1 - Data Ingestion, Loading and Cleaning
> In this first part of the two part series, we will handle the data side. We will fetch data from wikipedia, store in feather format and register the combined data on recochef. After that, we will perform EDA and extensive cleaning.

- toc: true
- badges: true
- comments: true
- categories: [Ingestion, EDA, DataCleaning]
- author: "<a href='https://towardsdatascience.com/how-can-you-tell-if-your-recommender-system-is-any-good-e4a6be02d9c2'>Daniel Saunders</a>"
- image:
<!-- #endregion -->

<!-- #region id="1rxKKMQB2Ygc" -->
## Data ingestion

Downloading data from Wikipedia using Wiki API and storing in Google drive. There are more than 11 millions records which would take 6-8 hours in single colab session, so used 10 colab workers to fetch all the data within 30-40 mins.
<!-- #endregion -->

```python id="EOyiO5ce2TcZ"
!git clone https://github.com/sparsh-ai/reco-wikirecs
%cd /content/reco-wikirecs/
!pip install -r requirements.txt
```

```python id="c_uAz5OS4sm3"
import yaml
import os
from wiki_pull import *
from itables.javascript import load_datatables
load_datatables()
```

```python id="iT-JOJiF3G8M"
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 521} id="tQPeT_sk7Mud" outputId="a248ea52-7dd1-4633-bd10-bc24d0201250"
get_sample_of_users(config['edit_lookback'], config['outfile'])
```

<!-- #region id="26kXx5-b3EUd" -->
### Start the ingestion

I ran the same code from start till this cell, in 10 different colab notebooks, changing start position. The design is a master-worker setup where 1 notebooks was the master one, and 9 are workers.

In master, start=0. In worker 1, start=5000. In worker 2, start=10000, and so on. This start value indicates the number of users. Since there are 54K users, each worker handled 5000 users on average.
<!-- #endregion -->

```python id="-vNLJxSLB2m8"
pull_edit_histories(
    config['outfile'],
    os.path.join(config['file_save_path'],config['edit_histories_file_pattern']),
    config['users_per_chunk'],
    config['earliest_timestamp'],
    start=0,
    )
```

<!-- #region id="K8K_erP24SfD" -->
## Data storage

During ingestion, we stored data in feather format parts. Now, we will combine the data and store in compressed parquet format. 

We will also register this data on recochef so that we can easily load it anywhere and also make it reusable for future use cases.
<!-- #endregion -->

```python id="ZsdBdQU4SlPy"
import os
import yaml
import pandas as pd
from pyarrow import feather
```

```python id="wXZfzk8rTYX6"
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
```

```python id="8k7gkgbdSYrF"
all_histories = []
for fname in os.listdir(config['file_save_path']):
    if 'feather' in  fname:
        all_histories.append(feather.read_feather(os.path.join(config['file_save_path'],fname)))
```

```python id="gMtXFbWLTpJT"
all_histories = pd.concat(all_histories, ignore_index=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Zg-ZRYUFVW33" outputId="76249d15-9f6e-45ef-8863-41cb6eef25d8"
all_histories.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="hU_SbzmmVeG1" outputId="1820caac-cb05-4c4a-a62d-064a88338d00"
all_histories.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="UYjNXm6bVy2o" outputId="17921817-c2e3-47b5-a75a-4ff45afe8ebc"
all_histories.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="Z6lerT-PVgXn" outputId="b2ba7815-305e-49b5-e527-8591fbcefe16"
all_histories.describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="mvU0BTqOVmXs" outputId="cea875e4-9b28-4a45-f68c-cf6f9fd15562"
all_histories.describe(include=['O'])
```

```python id="muyaq_HuVsBL"
all_histories.to_parquet('wikirecs.parquet.gzip', compression='gzip')
```

<!-- #region id="q_s-Fib4XF31" -->
> Note: Data is also registered with [recochef](https://github.com/sparsh-ai/recochef/blob/master/src/recochef/datasets/wikirecs.py) for easy access
<!-- #endregion -->

<!-- #region id="hMXlY1JF5Lvi" -->
## EDA and Data cleaning
<!-- #endregion -->

```python id="LLMOakVK7lZg"
!git clone https://github.com/sparsh-ai/reco-wikirecs
%cd /content/reco-wikirecs/
!pip install -r requirements.txt

!pip install -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="xX-hzqMQ5Odd"
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
### Data loading
<!-- #endregion -->

```python id="19-zA66p5Odf"
wikidata = WikiRecs()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ClFOpx4g5Odf" outputId="0a11cd60-7888-4f05-9a59-e857705c98d4"
df = wikidata.load_interactions()
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="1xeCAd1jcgvx" outputId="2d7289fc-a8bf-4f7a-f6c4-c26f550cfe3d"
df.head()
```

<!-- #region id="1Qu0d-0Qhg23" -->
### EDA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 435} id="KeeLRV8tfvOd" outputId="9b81f205-5de1-4460-a325-531eaf30eda3"
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

```python colab={"base_uri": "https://localhost:8080/"} id="Y36Q_2ZTgeMl" outputId="2ef045d2-d681-49c6-892b-f9e19fceb0de"
# Most edits by user
df.groupby(['USERID','USERNAME']).USERID.count().sort_values(ascending=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="DEWSXfV9g94l" outputId="3f5c41c6-8c59-40a7-d961-2d43ef9c02ee"
# Find the elbow in number of edits
plt.plot(df.groupby(['USERID','USERNAME']).USERID.count().sort_values(ascending=False).values)
# plt.ylim([0,20000])
```

```python colab={"base_uri": "https://localhost:8080/"} id="z-Yg51cShHdT" outputId="b1e55e48-974b-44f2-c224-7c6ea76a7df5"
# What are the most popular pages (edited by the most users)
page_popularity = df.drop_duplicates(subset=['TITLE','USERNAME']).groupby('TITLE').count().USERNAME.sort_values()
page_popularity.iloc[-1000:].iloc[::-1]
```

```python colab={"base_uri": "https://localhost:8080/"} id="T0p3yQorpFWs" outputId="3136f4a0-038c-4722-cf18-8cad5199fd3f"
df.sample().USERNAME
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="f5zITFxcozuU" outputId="92908c8e-2847-4d48-f1f9-3178a3bc5d53"
cols = ['userid', 'user', 'pageid', 'title',
       'timestamp', 'sizediff']

oneuser = get_edit_history(user="SanAnMan",
                            latest_timestamp="2021-07-08T22:02:09Z",
                            earliest_timestamp="2020-05-28T22:02:09Z")

oneuser = pd.DataFrame(oneuser).loc[:,cols]
oneuser
```

<!-- #region id="aF2RyBNxhiXv" -->
### Data cleaning
<!-- #endregion -->

<!-- #region id="Qu3Pz_tS-WMQ" -->
#### Remove consecutive edits and summarize runs
<!-- #endregion -->

```python id="7hWZQkiX-WMR" colab={"base_uri": "https://localhost:8080/"} outputId="98ac4ee0-37a6-44cb-d89e-e28d209ba3ee"
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
#### Remove top N most popular pages
<!-- #endregion -->

```python id="jOEyyMOs-WMU" colab={"base_uri": "https://localhost:8080/"} outputId="372cec11-d432-46f4-bf33-d52962a4a222"
# Get the top most popular pages
TOPN = 20
popularpages = df.drop_duplicates(subset=['TITLE','ITEMID','USERID']).groupby(['TITLE','ITEMID']).count().USERNAME.sort_values()[-TOPN:]
popularpages
```

```python id="uSz0qP6g-WMV" colab={"base_uri": "https://localhost:8080/"} outputId="f8dddfa2-c74d-4216-8492-6b6605b95a86"
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

```python id="YLAq2a7I-WMZ" colab={"base_uri": "https://localhost:8080/"} outputId="860cc588-196c-4d76-dc7e-67257d66a41a"
print("Number of unique page ids: {}".format(len(clean_df.ITEMID.unique())))
```

<!-- #region id="R8hqTBZZ-WMa" -->
#### Remove users with too many or too few edits
<!-- #endregion -->

```python id="WajFz7Bz-WMc"
MIN_EDITS = 5
MAX_EDITS = 10000
```

```python id="SnbOrhH9-WMd" colab={"base_uri": "https://localhost:8080/"} outputId="17467712-d2cd-4ee8-81fe-73901448d891"
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

```python id="WbfD8r0w-WMg" colab={"base_uri": "https://localhost:8080/"} outputId="0c2dc2e6-ab40-4d48-ff4d-4415f45f459a"
print("Length after removing users: {}".format(len(clean_df)))
```

<!-- #region id="RjNaQy7Y-WMi" -->
### Build lookup tables
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
### Build test and training set
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

```python colab={"base_uri": "https://localhost:8080/"} id="peWz8aTJmNGt" outputId="765d7fb4-2f3f-49cb-a2ad-b6675b7fa2db"
print("Length of test set: {}".format(len(histories_test)))
print("Length of dev set: {}".format(len(histories_dev)))
print("Length of training after removal of test: {}".format(len(histories_train)))
```

```python id="cYDo1XJM-WMr" colab={"base_uri": "https://localhost:8080/"} outputId="1c7d52b3-b616-4ca8-c810-4e0567a33a9d"
print("Number of pages in training set: {}".format(len(histories_train.ITEMID.unique())))
print("Number of users in training set: {}".format(len(histories_train.USERID.unique())))
print("Number of pages with > 1 user editing: {}".format(np.sum(histories_train.drop_duplicates(subset=['TITLE','USER']).groupby('TITLE').count().USER > 1)))
```

```python id="ht3O-0DL-WMx" colab={"base_uri": "https://localhost:8080/"} outputId="365a7edf-d948-4739-8705-5c1129ea27fe"
resurface_userids, discovery_userids = get_resurface_discovery(histories_train, histories_dev)

print("%d out of %d userids are resurfaced (%.1f%%)" % (len(resurface_userids), len(userids), 100*float(len(resurface_userids))/len(userids)))
print("%d out of %d userids are discovered (%.1f%%)" % (len(discovery_userids), len(userids), 100*float(len(discovery_userids))/len(userids)))
```

<!-- #region id="sgzzNkOxr8Z9" -->
### Build matrix for implicit collaborative filtering
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
### Saving artifacts
<!-- #endregion -->

```python id="cX7sQzl_nNx3"
save_pickle((p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t), 'lookup_tables.pickle')
save_pickle((userids, pageids), 'users_and_pages.pickle')
save_pickle((resurface_userids, discovery_userids), 'resurface_discovery_users.pickle')
save_pickle(implicit_matrix,'implicit_matrix.pickle')
```
