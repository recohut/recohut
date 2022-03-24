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

```python id="Dg8frDmMWhHA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628062221981, "user_tz": -330, "elapsed": 5887, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8c9441a2-08bc-4abf-be66-3522b82afd9f"
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

<!-- #region id="Hwg3020-GL4U" -->
### Dataloader for Interest modeling
<!-- #endregion -->

```python id="J_bq26aXEhBE"
import os
import csv
import pdb
import time
import pickle
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
from torch.utils.data.dataloader import default_collate
```

```python id="157QNwLiEy_k"
def toymd(time):
    return datetime.utcfromtimestamp(time)#.strftime('%Y-%m-%d')
```

```python id="2nguaw-hFJdh"
class Dataset(data.Dataset):

    def __init__(self, data):
        st = time.time()
        
        self.iids, self.labels, self.timediffs = [], [], []
        self.most_oldtime = None
        
        for row in data:
            self.iids.append(row[0])
            self.labels.append(row[1])
            self.timediffs.append(row[2:])
            
        self.iids = np.array(self.iids)
        self.timediffs = np.array(self.timediffs).astype(int)
        self.labels = (np.array(self.labels) == 'True').astype(int) 
        
        print('Data building time : %.1fs' % (time.time()-st))
        
    def __getitem__(self, index):
        return self.iids[index], self.timediffs[index], self.labels[index]
    
    def __len__(self):
        """Returns the total number of user-item pairs."""
        return len(self.timediffs)
```

```python id="uiad4e3BFN_0"
def build_loader(eachdata, batch_size, shuffle=True, num_workers=0):
    
    def my_collate(batch):
        batch = [i for i in filter(lambda x:x is not None, batch)]
        return default_collate(batch)
    
    """Builds and returns Dataloader."""
    dataset = Dataset(eachdata)
    
    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate)

    return data_loader  
```

```python id="2R_iv7xsE3Yi"
def build_data_directly(dpath, period, binsize):
    def toymd(time):
        return datetime.utcfromtimestamp(time)
    
    def build_data(true_items, item_feature):
        output = []
        for i in item_feature:
            feature = item_feature[i]
            instance = [i] + [bool(i in true_items)] + list(feature) # [iid, label, features]
            output.append(instance)    
        return np.array(output)
    
    def get_item_feature(data):
        times = data[:,-1].astype(float).astype(int)
        mintime, maxtime = toymd(min(times)), toymd(max(times))

        # Binning training time (D_f) with fixed-sized bins
        timedelta = relativedelta(weeks=binsize)
        bins = np.array([mintime + timedelta*i for i in range(1000) # quick implementation
                         if mintime + timedelta*i < maxtime + timedelta*0])

        # Build features from data
        idict = {}
        for u, i, r, t in data:
            if i not in idict: idict[i] = []
            idict[i].append(toymd(int(float(t))))

        # Build features for each item
        item_feature = {}
        for i in idict:
            times = np.array(idict[i])

            # Transform times into frequency bins
            binned_times = []
            for t in times:
                binidx = np.where(bins <= t)[0][-1]
                each_binfeature = np.zeros(len(bins))
                each_binfeature[binidx] = 1
                binned_times.append(each_binfeature)
            binned_times = np.array(binned_times).sum(axis=0).astype(int)

            item_feature[i] = binned_times
            
        return item_feature

    rawtrn = np.array([l for l in csv.reader(open(dpath+'train.csv'))])
    rawvld = np.array([l for l in csv.reader(open(dpath+'valid.csv'))])
    rawtst = np.array([l for l in csv.reader(open(dpath+'test.csv'))])
    
    times_trn = rawtrn[:,-1].astype(int)
    
    # Split data by period (unit: week)
    # [trn_start - trnfront - vld_start - tst_start - tst_end]
    trnfront_time = times_trn.max() - 60 * 60 * 24 * 7 * period 
    trnfront_idx = np.where(times_trn < trnfront_time)[0][-1]
    trn_start_time = int(float(times_trn[0])) # -1 denotes the time index
    trnfront_start_time = int(float(rawtrn[trnfront_idx][-1]))
    vld_start_time = int(float(rawvld[0][-1]))
    tst_start_time = int(float(rawtst[0][-1]))
    tst_end_time = int(float(rawtst[-1][-1]))
    
    print('\n📋 Data loaded from: {}\n'.format(dpath))

    print('Trn start time:\t{}'.format(toymd(trn_start_time)))
    print('Trn front time:\t{}'.format(toymd(trnfront_start_time)))
    print('Vld start time:\t{}'.format(toymd(vld_start_time)))
    print('Tst start time:\t{}'.format(toymd(tst_start_time)))
    print('Tst end time:\t{}'.format(toymd(tst_end_time)))
    
    trn_4feature = rawtrn[:trnfront_idx]
    feature_trn = get_item_feature(trn_4feature) # features for training
    feature_eval = get_item_feature(rawtrn) # features for evaluation (to get ISS for training RS)
    
    trn_4label = rawtrn[trnfront_idx:] # D_b
    
    trndata = build_data(set(trn_4label[:,1]), feature_trn)
    vlddata = build_data(set(rawvld[:,1]), feature_eval)
    tstdata = build_data(set(rawtst[:,1]), feature_eval)
    
    return trndata, vlddata, tstdata
```

```python id="Ifd1oGXyFnzl"
class DataLoader:
    def __init__(self, opt):
        self.dpath = opt.dataset_path + '/'
        self.batch_size = opt.batch_size
        
        trndata, vlddata, tstdata = build_data_directly(self.dpath, opt.period, opt.binsize)        
        
        self.trn_loader = build_loader(trndata, opt.batch_size, shuffle=True)
        self.vld_loader = build_loader(vlddata, opt.batch_size, shuffle=False)
        self.tst_loader = build_loader(tstdata, opt.batch_size, shuffle=False)
        
        print(("train/val/test/ divided by batch size {:d}/{:d}/{:d}".format(len(self.trn_loader), len(self.vld_loader),len(self.tst_loader))))
        print("==================================================================================")
            
    def get_loaders(self):
        return self.trn_loader, self.vld_loader, self.tst_loader
    
    def get_embedding(self):
        return self.input_embedding
```

<!-- #region id="MOI9KSIIHgqp" -->
### Unit testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kb_GXlmSH1Jl" executionInfo={"status": "ok", "timestamp": 1628063380215, "user_tz": -330, "elapsed": 628, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d1ae4a8f-30d3-492e-b4c2-d2a212303bb1"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amazon_tools', type=str)    
parser.add_argument('--period', default=16, type=float)
parser.add_argument('--binsize', default=8, type=int)
parser.add_argument('--learning_rate', default=1e-2, type=float)
parser.add_argument('--l2reg', default=1e-4, type=float)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)    
parser.add_argument('--hidden_dim', default=64, type=int)    
parser.add_argument('--pos_weight', default=1e-2, type=float)   
parser.add_argument('--gpu', default=3, type=int)       

opt = parser.parse_args(args={})
dataset_path = './data/silver/{}'.format(opt.dataset)    

opt.dataset_path = dataset_path
opt
```

```python colab={"base_uri": "https://localhost:8080/"} id="50Qr_uYHHgJX" executionInfo={"status": "ok", "timestamp": 1628063464490, "user_tz": -330, "elapsed": 9043, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6b2fe302-685c-4a44-f2b2-3e6fa1dde97a"
from collections import Counter

data_loader = DataLoader(opt)
trn_loader, vld_loader, tst_loader = data_loader.get_loaders()

trnlen = trn_loader.dataset.timediffs.shape[1]        

print('TRN labels: {}'.format(Counter(trn_loader.dataset.labels)))
print('VLD labels: {}'.format(Counter(vld_loader.dataset.labels)))
print('TST labels: {}'.format(Counter(tst_loader.dataset.labels)))
```

<!-- #region id="Co36ZXYgGGIw" -->
### Dataloader for recommendation modeling
<!-- #endregion -->

```python id="RiDKoWXsGqQK"
import os
import pdb
import time
import torch
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data.dataloader import default_collate

random.seed(2020)
```

```python id="dQQDpx8YGTj5"
class ML_Dataset(data.Dataset):
    
    def build_consumption_history(self, uir):
        # Build a dictionary for user: items consumed by the user
        uir = uir.astype(int)
        uidict = {}
        allitems = set()
        for u, i, _ in uir:
            if u not in uidict: uidict[u] = set()
            uidict[u].add(i)
            allitems.add(i)
            
        self.ui_cand_dict = {}    
        for u in uidict:
            self.ui_cand_dict[u] = np.array(list(allitems - uidict[u]))
        
        return uidict, allitems
        
    def __init__(self, path, trn_numneg):
        dpath = '/'.join(path.split('/')[:-1])
        if dpath[-1] != '/': dpath += '/'
        dtype = path.split('/')[-1].split('.')[0]
        
        st = time.time()        
        
        if dtype == 'train': self.numneg = trn_numneg
        self.uir = np.load(path)

        if dtype == 'train':             
            self.uir[:,-1] = 1 # Mark explicit feedback as implicit feedback

            self.first = self.uir[:,0].astype(int)
            self.second = self.uir[:,1].astype(int)
            self.third = np.zeros(self.uir.shape[0]) # This will be replaced in 'train_collate'
            
            self.numuser = len(set(self.uir[:,0].astype(int)))
            self.numitem = len(set(self.uir[:,1].astype(int)))
            
            self.uidict, self.allitems = self.build_consumption_history(self.uir)
            
        elif dtype == 'valid' or dtype == 'test':             
            # Build validation data for ranking evaluation
            newuir = []
            for row in self.uir:
                user = row[0]
                true_item = row[1]
                newuir.append([user, true_item, 1]) # a true consumption
                for item in row[2:]: newuir.append([user, item, 0]) # negative candidates
            self.uir = np.array(newuir) # User, Item, Rating
        
            self.first, self.second, self.third = self.uir[:,0], self.uir[:,1], self.uir[:,2]
        
        
        print('Data building time : %.1fs' % (time.time()-st))

    def __getitem__(self, index):
        # Training: [user, positive, negative]
        # Testing: [user, canidate item, label] 
        return self.first[index], self.second[index], self.third[index]
    
    def __len__(self):
        """Returns the total number of user-item pairs."""
        return len(self.first)
    
    
    def train_collate(self, batch):
        # Input: [user, postive item, dummy]
        # Output: [user, positive item, negative item]
        batch = [i for i in filter(lambda x:x is not None, batch)]
        
        # Negative sampling for each batch
        outputs = []
        for u, pi, dummy in batch:
            rand_idx = np.random.randint(len(self.ui_cand_dict[u]), size=self.numneg)
            neg_items = self.ui_cand_dict[u][rand_idx]
            
            for ni in neg_items: 
                outputs.append([u, pi, ni])
            
        return default_collate(outputs)      
```

```python id="4vypXyYKGmDH"
def test_collate(batch):
    batch = [i for i in filter(lambda x:x is not None, batch)]
    return default_collate(batch)
```

```python id="lKsWcrQpGkdH"
def get_each_loader(data_path, batch_size, trn_negnum, shuffle=True, num_workers=0):
    """Builds and returns Dataloader."""
    
    dataset = ML_Dataset(data_path, trn_negnum)
    
    if data_path.endswith('train.npy') == True:
        collate = dataset.train_collate
    else:
        collate = test_collate

    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate)

    return data_loader
```

```python id="9PME29ToGdMJ"
class DataLoader:
    def __init__(self, opt):
        self.dpath = opt.dataset_path + '/'
        self.batch_size = opt.batch_size
        self.trn_numneg = opt.numneg
        
        self.trn_loader, self.vld_loader, self.tst_loader = self.get_loaders_for_metric_learning(self.trn_numneg)
    
        print(("train/val/test/ divided by batch size {:d}/{:d}/{:d}".format(len(self.trn_loader), len(self.vld_loader),len(self.tst_loader))))
        print("=" * 80)
        
    def get_loaders_for_metric_learning(self, trn_numneg):
        print("\n📋 Loading data...\n")
        trn_loader = get_each_loader(self.dpath+'train.npy', self.batch_size, trn_numneg, shuffle=True)
        print('\tTraining data loaded')
        
        vld_loader = get_each_loader(self.dpath+'valid.npy', self.batch_size, trn_numneg, shuffle=False)
        print('\tValidation data loaded')
        
        tst_loader = get_each_loader(self.dpath+'test.npy', self.batch_size, trn_numneg, shuffle=False)
        print('\tTest data loaded')
        
        return trn_loader, vld_loader, tst_loader
    
    def get_loaders(self):
        return self.trn_loader, self.vld_loader, self.tst_loader
    
    def get_embedding(self):
        return self.input_embedding
```

<!-- #region id="AEAkK5qGM8wF" -->
### Unit testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6-JxFH-aM8wW" executionInfo={"status": "ok", "timestamp": 1628064656757, "user_tz": -330, "elapsed": 579, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2c0e8d79-362d-44f0-9163-eaf1babc2934"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amazon_tools', type=str)    
parser.add_argument('--batch_size', default=4096, type=int)      
parser.add_argument('--numneg', default=10, type=int)

opt = parser.parse_args(args={})
dataset_path = './data/gold/{}'.format(opt.dataset)    

opt.dataset_path = dataset_path
opt
```

```python colab={"base_uri": "https://localhost:8080/"} id="e7a3BiecM8wY" executionInfo={"status": "ok", "timestamp": 1628064752679, "user_tz": -330, "elapsed": 23655, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2359660a-6929-4990-a950-99aceb64792f"
data_loader = DataLoader(opt)

trn_loader, vld_loader, tst_loader = data_loader.get_loaders()

opt.numuser = trn_loader.dataset.numuser
opt.numitem = trn_loader.dataset.numitem
```

<!-- #region id="uWHShmo2OOa7" -->
## Exporting the methods
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nBdOOntLOQ8v" executionInfo={"status": "ok", "timestamp": 1628064889575, "user_tz": -330, "elapsed": 683, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b7c5032-6bd9-401d-a34e-bdbc74db67f8"
%%writefile ./code/dataloader_interest.py
import os
import csv
import pdb
import time
import pickle
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
from torch.utils.data.dataloader import default_collate


def toymd(time):
    return datetime.utcfromtimestamp(time)#.strftime('%Y-%m-%d')


class Dataset(data.Dataset):

    def __init__(self, data):
        st = time.time()
        
        self.iids, self.labels, self.timediffs = [], [], []
        self.most_oldtime = None
        
        for row in data:
            self.iids.append(row[0])
            self.labels.append(row[1])
            self.timediffs.append(row[2:])
            
        self.iids = np.array(self.iids)
        self.timediffs = np.array(self.timediffs).astype(int)
        self.labels = (np.array(self.labels) == 'True').astype(int) 
        
        print('Data building time : %.1fs' % (time.time()-st))
        
    def __getitem__(self, index):
        return self.iids[index], self.timediffs[index], self.labels[index]
    
    def __len__(self):
        """Returns the total number of user-item pairs."""
        return len(self.timediffs)


def build_loader(eachdata, batch_size, shuffle=True, num_workers=0):
    
    def my_collate(batch):
        batch = [i for i in filter(lambda x:x is not None, batch)]
        return default_collate(batch)
    
    """Builds and returns Dataloader."""
    dataset = Dataset(eachdata)
    
    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate)

    return data_loader  


def build_data_directly(dpath, period, binsize):
    def toymd(time):
        return datetime.utcfromtimestamp(time)
    
    def build_data(true_items, item_feature):
        output = []
        for i in item_feature:
            feature = item_feature[i]
            instance = [i] + [bool(i in true_items)] + list(feature) # [iid, label, features]
            output.append(instance)    
        return np.array(output)
    
    def get_item_feature(data):
        times = data[:,-1].astype(float).astype(int)
        mintime, maxtime = toymd(min(times)), toymd(max(times))

        # Binning training time (D_f) with fixed-sized bins
        timedelta = relativedelta(weeks=binsize)
        bins = np.array([mintime + timedelta*i for i in range(1000) # quick implementation
                         if mintime + timedelta*i < maxtime + timedelta*0])

        # Build features from data
        idict = {}
        for u, i, r, t in data:
            if i not in idict: idict[i] = []
            idict[i].append(toymd(int(float(t))))

        # Build features for each item
        item_feature = {}
        for i in idict:
            times = np.array(idict[i])

            # Transform times into frequency bins
            binned_times = []
            for t in times:
                binidx = np.where(bins <= t)[0][-1]
                each_binfeature = np.zeros(len(bins))
                each_binfeature[binidx] = 1
                binned_times.append(each_binfeature)
            binned_times = np.array(binned_times).sum(axis=0).astype(int)

            item_feature[i] = binned_times
            
        return item_feature

    rawtrn = np.array([l for l in csv.reader(open(dpath+'train.csv'))])
    rawvld = np.array([l for l in csv.reader(open(dpath+'valid.csv'))])
    rawtst = np.array([l for l in csv.reader(open(dpath+'test.csv'))])
    
    times_trn = rawtrn[:,-1].astype(int)
    
    # Split data by period (unit: week)
    # [trn_start - trnfront - vld_start - tst_start - tst_end]
    trnfront_time = times_trn.max() - 60 * 60 * 24 * 7 * period 
    trnfront_idx = np.where(times_trn < trnfront_time)[0][-1]
    trn_start_time = int(float(times_trn[0])) # -1 denotes the time index
    trnfront_start_time = int(float(rawtrn[trnfront_idx][-1]))
    vld_start_time = int(float(rawvld[0][-1]))
    tst_start_time = int(float(rawtst[0][-1]))
    tst_end_time = int(float(rawtst[-1][-1]))
    
    print('\n📋 Data loaded from: {}\n'.format(dpath))

    print('Trn start time:\t{}'.format(toymd(trn_start_time)))
    print('Trn front time:\t{}'.format(toymd(trnfront_start_time)))
    print('Vld start time:\t{}'.format(toymd(vld_start_time)))
    print('Tst start time:\t{}'.format(toymd(tst_start_time)))
    print('Tst end time:\t{}'.format(toymd(tst_end_time)))
    
    trn_4feature = rawtrn[:trnfront_idx]
    feature_trn = get_item_feature(trn_4feature) # features for training
    feature_eval = get_item_feature(rawtrn) # features for evaluation (to get ISS for training RS)
    
    trn_4label = rawtrn[trnfront_idx:] # D_b
    
    trndata = build_data(set(trn_4label[:,1]), feature_trn)
    vlddata = build_data(set(rawvld[:,1]), feature_eval)
    tstdata = build_data(set(rawtst[:,1]), feature_eval)
    
    return trndata, vlddata, tstdata


class DataLoader:
    def __init__(self, opt):
        self.dpath = opt.dataset_path + '/'
        self.batch_size = opt.batch_size
        
        trndata, vlddata, tstdata = build_data_directly(self.dpath, opt.period, opt.binsize)        
        
        self.trn_loader = build_loader(trndata, opt.batch_size, shuffle=True)
        self.vld_loader = build_loader(vlddata, opt.batch_size, shuffle=False)
        self.tst_loader = build_loader(tstdata, opt.batch_size, shuffle=False)
        
        print(("train/val/test/ divided by batch size {:d}/{:d}/{:d}".format(len(self.trn_loader), len(self.vld_loader),len(self.tst_loader))))
        print("==================================================================================")
            
    def get_loaders(self):
        return self.trn_loader, self.vld_loader, self.tst_loader
    
    def get_embedding(self):
        return self.input_embedding
```

```python colab={"base_uri": "https://localhost:8080/"} id="OvWExDTSOstr" executionInfo={"status": "ok", "timestamp": 1628064979872, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b0a60847-522a-4350-b24a-40a42c3e2f92"
%%writefile ./code/dataloader_recommendation.py
import os
import pdb
import time
import torch
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data.dataloader import default_collate

random.seed(2020)


class ML_Dataset(data.Dataset):
    
    def build_consumption_history(self, uir):
        # Build a dictionary for user: items consumed by the user
        uir = uir.astype(int)
        uidict = {}
        allitems = set()
        for u, i, _ in uir:
            if u not in uidict: uidict[u] = set()
            uidict[u].add(i)
            allitems.add(i)
            
        self.ui_cand_dict = {}    
        for u in uidict:
            self.ui_cand_dict[u] = np.array(list(allitems - uidict[u]))
        
        return uidict, allitems
        
    def __init__(self, path, trn_numneg):
        dpath = '/'.join(path.split('/')[:-1])
        if dpath[-1] != '/': dpath += '/'
        dtype = path.split('/')[-1].split('.')[0]
        
        st = time.time()        
        
        if dtype == 'train': self.numneg = trn_numneg
        self.uir = np.load(path)

        if dtype == 'train':             
            self.uir[:,-1] = 1 # Mark explicit feedback as implicit feedback

            self.first = self.uir[:,0].astype(int)
            self.second = self.uir[:,1].astype(int)
            self.third = np.zeros(self.uir.shape[0]) # This will be replaced in 'train_collate'
            
            self.numuser = len(set(self.uir[:,0].astype(int)))
            self.numitem = len(set(self.uir[:,1].astype(int)))
            
            self.uidict, self.allitems = self.build_consumption_history(self.uir)
            
        elif dtype == 'valid' or dtype == 'test':             
            # Build validation data for ranking evaluation
            newuir = []
            for row in self.uir:
                user = row[0]
                true_item = row[1]
                newuir.append([user, true_item, 1]) # a true consumption
                for item in row[2:]: newuir.append([user, item, 0]) # negative candidates
            self.uir = np.array(newuir) # User, Item, Rating
        
            self.first, self.second, self.third = self.uir[:,0], self.uir[:,1], self.uir[:,2]
        
        
        print('Data building time : %.1fs' % (time.time()-st))

    def __getitem__(self, index):
        # Training: [user, positive, negative]
        # Testing: [user, canidate item, label] 
        return self.first[index], self.second[index], self.third[index]
    
    def __len__(self):
        """Returns the total number of user-item pairs."""
        return len(self.first)
    
    
    def train_collate(self, batch):
        # Input: [user, postive item, dummy]
        # Output: [user, positive item, negative item]
        batch = [i for i in filter(lambda x:x is not None, batch)]
        
        # Negative sampling for each batch
        outputs = []
        for u, pi, dummy in batch:
            rand_idx = np.random.randint(len(self.ui_cand_dict[u]), size=self.numneg)
            neg_items = self.ui_cand_dict[u][rand_idx]
            
            for ni in neg_items: 
                outputs.append([u, pi, ni])
            
        return default_collate(outputs)      


def test_collate(batch):
    batch = [i for i in filter(lambda x:x is not None, batch)]
    return default_collate(batch)


def get_each_loader(data_path, batch_size, trn_negnum, shuffle=True, num_workers=0):
    """Builds and returns Dataloader."""
    
    dataset = ML_Dataset(data_path, trn_negnum)
    
    if data_path.endswith('train.npy') == True:
        collate = dataset.train_collate
    else:
        collate = test_collate

    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate)

    return data_loader


class DataLoader:
    def __init__(self, opt):
        self.dpath = opt.dataset_path + '/'
        self.batch_size = opt.batch_size
        self.trn_numneg = opt.numneg
        
        self.trn_loader, self.vld_loader, self.tst_loader = self.get_loaders_for_metric_learning(self.trn_numneg)
    
        print(("train/val/test/ divided by batch size {:d}/{:d}/{:d}".format(len(self.trn_loader), len(self.vld_loader),len(self.tst_loader))))
        print("=" * 80)
        
    def get_loaders_for_metric_learning(self, trn_numneg):
        print("\n📋 Loading data...\n")
        trn_loader = get_each_loader(self.dpath+'train.npy', self.batch_size, trn_numneg, shuffle=True)
        print('\tTraining data loaded')
        
        vld_loader = get_each_loader(self.dpath+'valid.npy', self.batch_size, trn_numneg, shuffle=False)
        print('\tValidation data loaded')
        
        tst_loader = get_each_loader(self.dpath+'test.npy', self.batch_size, trn_numneg, shuffle=False)
        print('\tTest data loaded')
        
        return trn_loader, vld_loader, tst_loader
    
    def get_loaders(self):
        return self.trn_loader, self.vld_loader, self.tst_loader
    
    def get_embedding(self):
        return self.input_embedding
```

```python colab={"base_uri": "https://localhost:8080/"} id="cZQxNJELPJAE" executionInfo={"status": "ok", "timestamp": 1628065019801, "user_tz": -330, "elapsed": 618, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="476edcb5-999c-471b-a3e9-180c69757818"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="ojGU36JOPJjD" executionInfo={"status": "ok", "timestamp": 1628065057555, "user_tz": -330, "elapsed": 1412, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d45f82a-91a9-44ab-999f-5f33e6769298"
!git add . && git commit -m 'ADD code dataloaders for interest modeling and recommendations' && git push origin main
```
