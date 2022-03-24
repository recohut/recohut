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

```python id="Dg8frDmMWhHA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628062000066, "user_tz": -330, "elapsed": 5601, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="afe2b1d6-9491-4bf1-c475-7e8dec51e2b1"
import os
project_name = "reco-tut-cris"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

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

```python id="KnczXQvP4Se7" executionInfo={"status": "ok", "timestamp": 1628062000068, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import sys
import csv
import pdb
import copy
import random
import numpy as np
import itertools
from collections import Counter
```

```python id="BCSPhBr24VDC" executionInfo={"status": "ok", "timestamp": 1628062000069, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def replace_id2idx(trn, vld, tst):
    
    def build_dict(category):
        category = list(set(category))

        cate_dict = {}
        for i, c in enumerate(category): cate_dict[c] = i
        return cate_dict

    def id2idx(uir, udict, idict): # Convert IDs in string into IDs in numbers
        newuir = []
        for i in range(len(uir)):
            user, item, rating, _ = uir[i] # Fourth element is a time stamp for the interaction
            newuir.append([udict[user], idict[item], rating])
        return newuir

    trn_users = [i[0] for i in trn] 
    trn_items = [i[1] for i in trn] 
    
    user_dict = build_dict(trn_users)
    item_dict = build_dict(trn_items)
    
    trn = id2idx(trn, user_dict, item_dict)
    vld = id2idx(vld, user_dict, item_dict)
    tst = id2idx(tst, user_dict, item_dict)
    
    return trn, vld, tst, user_dict, item_dict
```

```python id="Zxo2iVv55E6v" executionInfo={"status": "ok", "timestamp": 1628062000070, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def load_raw_data(fn):
    print('Load ' + fn)
    rawdata = [l for l in csv.reader(open(fn))]
    return rawdata
```

```python id="JEYZtQ1C_joB" executionInfo={"status": "ok", "timestamp": 1628062000071, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def find_negatives(dataset):
    NUMNEG = 100
    
    trn, vld, tst = dataset
    
    allitems = set([i[1] for i in trn])
    
    uidict = {} # {u: [items consumed by user u]}
    for i in range(len(trn)):
        user, item, rating = trn[i]
        if user not in uidict: uidict[user] = []
        uidict[user].append(item)
    
    for i in range(len(vld)):
        user, item, _ = vld[i]
            
        useritems = set(uidict[user] + [item]) # Target item and a user's consumed items
        negative_items = random.sample(list(allitems - useritems), NUMNEG)
        
        vld[i] = vld[i][:-1] + negative_items # Append negative items for evaluation
    
    for i in range(len(tst)):
        user, item, _ = tst[i]
        
        useritems = set(uidict[user] + [item])
        negative_items = random.sample(list(allitems - useritems), NUMNEG) 
        
        tst[i] = tst[i][:-1] + negative_items
    
    return trn, vld, tst
```

```python colab={"base_uri": "https://localhost:8080/"} id="zBZ8hzu6_-fn" executionInfo={"status": "ok", "timestamp": 1628062001909, "user_tz": -330, "elapsed": 1855, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c1aabfaf-2e24-4d7b-a9c4-ddbc3de2bf75"
data_path = './data/silver/amazon_tools/'

print('\n🧰 Building a dataset for training the recommender system \n')

for fn in os.listdir(data_path):
    if 'train' in fn: trndata_name = data_path+fn
    if 'valid' in fn: vlddata_name = data_path+fn
    if 'test' in fn: tstdata_name = data_path+fn

# Load datasets and review features from csv format
trndata = load_raw_data(trndata_name)
vlddata = load_raw_data(vlddata_name)
tstdata = load_raw_data(tstdata_name)

trndata, org_vlddata, org_tstdata, user2id_dict, item2id_dict = replace_id2idx(trndata, vlddata, tstdata)

trndat, vlddata, tstdata = find_negatives([trndata, copy.deepcopy(org_vlddata), copy.deepcopy(org_tstdata)])

print('\nTRAIN:{}\tVALID:{}\tTEST:{}'.format(len(trndata), len(vlddata), len(tstdata)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="L2aMVz1e_qSS" executionInfo={"status": "ok", "timestamp": 1628062002487, "user_tz": -330, "elapsed": 583, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="000a5ca9-441f-4ae1-c6ab-46109397cc11"
print('\n📂 Starting to save datasets')
data_path = './data/gold/amazon_tools/'
if not os.path.exists(data_path): os.makedirs(data_path)

np.save(open(data_path+'train.npy','wb'), np.array(trndata).astype(float).astype(int))
np.save(open(data_path+'valid.npy','wb'), np.array(vlddata).astype(float).astype(int))
np.save(open(data_path+'test.npy','wb'), np.array(tstdata).astype(float).astype(int))
np.save(open(data_path+'user_dict.npy','wb'), user2id_dict)
np.save(open(data_path+'item_dict.npy','wb'), item2id_dict)

print('\nDatasets saved to the data directory: {}\n'.format(data_path))
```

```python colab={"base_uri": "https://localhost:8080/"} id="ApO_qIsyDjit" executionInfo={"status": "ok", "timestamp": 1628062002493, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="142529ec-5ddd-4f7b-fbf3-f95156e6e98d"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="L_JAn16oDpOW" executionInfo={"status": "ok", "timestamp": 1628062025162, "user_tz": -330, "elapsed": 3888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e045afae-01b0-4f22-f534-2ed1deeb50b2"
!git add . && git commit -m 'ADD data in gold layer amazon tools' && git push origin main
```
