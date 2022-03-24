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

```python id="Dg8frDmMWhHA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628159429616, "user_tz": -330, "elapsed": 5178, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="784ad1f2-0e67-41e4-d135-2f7efe3a9e7e"
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

```python id="ziquRWpm0Q48" executionInfo={"status": "ok", "timestamp": 1628159429620, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0, './code')
```

```python id="PPu7FMc33e0T" executionInfo={"status": "ok", "timestamp": 1628159636766, "user_tz": -330, "elapsed": 609, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import pdb
import time
import math
import copy
import numpy as np
import argparse
import random
from numpy import std as STD
from numpy import average as AVG
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dataloader_recommendation import DataLoader
```

```python colab={"base_uri": "https://localhost:8080/"} id="mjWBFr2W3pq4" executionInfo={"status": "ok", "timestamp": 1628159645541, "user_tz": -330, "elapsed": 1430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d4899b15-a307-4258-9dd2-a884a4711c2a"
torch.set_num_threads(4)

random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
```

```python id="dyN_MX5z4K1s" executionInfo={"status": "ok", "timestamp": 1628159655032, "user_tz": -330, "elapsed": 700, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def HitRatio(ranklist):
    return bool(1 in ranklist)    

def NDCG(ranklist):
    for i, label in enumerate(ranklist):
        if label == 1: # True consumption
            return math.log(2) / math.log(i+2)
    return 0
```

```python id="yrEuTt7X4NZ7" executionInfo={"status": "ok", "timestamp": 1628159664584, "user_tz": -330, "elapsed": 618, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def return_perf(predictions, ipdict):
    predictions = predictions.reshape(-1, 101, 4) # 1 positive and 100 negatives
        
    topks = [2,5,10,20]
    
    hrs, ndcgs = {}, {}
    for tk in topks:
        hrs[tk] = 0
        ndcgs[tk] = 0
    
    for row in predictions: 
        inst = row[:, 1:] # [i, score, label]
        # To set wrong if all predictions are the same,
        # move the positive item to the end of the list.
        inst[[0, -1]] = inst[[-1, 0]] 
        inst = inst[inst[:,1].argsort()] # items with small distance will be at upper position
        
        for tk in topks:
            topk_labels = inst[:tk, -1]
            hrs[tk] += HitRatio(topk_labels)
            ndcgs[tk] += NDCG(topk_labels)
    
    numinst = predictions.shape[0]
    
    for tk in topks:
        hrs[tk] /= numinst
        ndcgs[tk] /= numinst
        
    return hrs, ndcgs
```

```python id="LfOXXf7I3-JG" executionInfo={"status": "ok", "timestamp": 1628159670215, "user_tz": -330, "elapsed": 878, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def _cal_ranking_measures(loader, model, opt, ipdict):
    predictions = np.array([])
    all_output, all_label = [], []
    all_uid, all_iid = [], []  
    all_interest = []
    
    for i, batch_data in enumerate(loader):
        batch_data = [bd.cuda() for bd in batch_data]
        
        user, item, label = batch_data
        dist, interest = model([user, item])

        all_interest.append(interest)
            
        all_output.append(dist)
        all_label.append(label)
        all_uid.append(user)
        all_iid.append(item)

    all_output = torch.cat(all_output).cpu().data.numpy()
    all_label = torch.cat(all_label).cpu().data.numpy()
    all_uid = torch.cat(all_uid).cpu().data.numpy()
    all_iid = torch.cat(all_iid).cpu().data.numpy()    
    
    if len(all_interest) != 0: all_interest = torch.cat(all_interest).cpu().data.numpy()
        
    total_output = all_output + opt.gamma * all_interest        
    predictions = np.array([all_uid, all_iid, total_output, all_label]).T
    hrs, ndcgs = return_perf(predictions, ipdict)
    
    return hrs, ndcgs
```

```python id="0_eSJPZ83_wE" executionInfo={"status": "ok", "timestamp": 1628159670789, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def cal_measures(loader, model, opt, ipdict):
    model.eval()    
    
    results = _cal_ranking_measures(loader, model, opt, ipdict)
    
    model.train()
    
    return results
```

```python id="GOa3xass3XSz" executionInfo={"status": "ok", "timestamp": 1628159679837, "user_tz": -330, "elapsed": 608, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class CRIS(nn.Module):
    def __init__(self, opt):
        super(CRIS, self).__init__()
        
        self.ebd_size = opt.K
        self.numuser = opt.numuser
        self.numitem = opt.numitem

        NUM_PROTOTYPE = 2 # prototype C and S
        
        self.ebd_user = nn.Embedding(self.numuser+1, self.ebd_size).cuda()
        self.ebd_item = nn.Embedding(self.numitem+1, self.ebd_size).cuda()    
        self.ebd_prototype = nn.Embedding(NUM_PROTOTYPE, self.ebd_size).cuda() 
        
        nn.init.xavier_normal_(self.ebd_user.weight)
        nn.init.xavier_normal_(self.ebd_item.weight)
        nn.init.xavier_normal_(self.ebd_prototype.weight)

        self.consumption_idx = torch.zeros(1).long().cuda()
        self.interest_idx = torch.ones(1).long().cuda()
        
    def forward(self, batch_data):
        user, item = batch_data
        
        consumption = self.ebd_prototype(self.consumption_idx)
        interest = self.ebd_prototype(self.interest_idx)
        
        embedded_user = self.ebd_user(user)
        embedded_item = self.ebd_item(item)
        
        ui_feature = embedded_user + embedded_item
        
        c_dist = F.pairwise_distance(consumption, ui_feature, 2)
        i_dist = F.pairwise_distance(interest, ui_feature, 2)
        
        return c_dist, i_dist
```

```python id="E-ExLjxP4cCw" executionInfo={"status": "ok", "timestamp": 1628159816967, "user_tz": -330, "elapsed": 632, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# copying interest probability file from silver to gold layer
!cp ./data/silver/interest_prob ./data/gold/amazon_tools/
```

```python id="MZSuilWf5BMI" executionInfo={"status": "ok", "timestamp": 1628159877829, "user_tz": -330, "elapsed": 827, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class Instructor:
    def __init__(self, opt):
        self.opt = opt
        
        self.data_loader = DataLoader(self.opt)

        self.trn_loader, self.vld_loader, self.tst_loader = self.data_loader.get_loaders()
        
        opt.numuser = self.trn_loader.dataset.numuser
        opt.numitem = self.trn_loader.dataset.numitem
        self.model = self.opt.model_class(self.opt).cuda()
        
        self._print_args()
        
    def train(self):
        # Load ISSs of items
        iid, prob = np.load(opt.dataset_path+'/interest_prob')
        prob = prob.astype(float)

        try:
            itdict = np.load(opt.dataset_path+'/item_dict.npy').item()
        except:
            itdict = np.load(opt.dataset_path+'/item_dict.npy',allow_pickle=True).item() # for numpy 0.17+

        ipdict = {} # {item ID: its ISS}
        for i in range(len(iid)):
            itemid = iid[i]
            if itemid in itdict:
                ipdict[itdict[itemid]] = prob[i]
        
        newtime = round(time.time())        
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.learning_rate)
            
        best_score = -1 
        best_topHits, best_topNdcgs = None, None
        batch_loss = 0
        c = 0 # to check early stopping
        
        self.clip_max_user = torch.FloatTensor([1.0]).cuda()
        self.clip_max_item = torch.FloatTensor([1.0]).cuda()
        self.clip_max_pro = torch.FloatTensor([1.0]).cuda()

        for epoch in range(self.opt.num_epoch):
            st = time.time()
    
            for i, batch_data in enumerate(self.trn_loader):
            
                # Unit-sphere restriction
                user_weight = self.model.ebd_user.weight.data
                user_weight.div_(torch.max(torch.norm(user_weight, 2, 1, True),
                                           self.clip_max_user).expand_as(user_weight))

                item_weight = self.model.ebd_item.weight.data
                item_weight.div_(torch.max(torch.norm(item_weight, 2, 1, True),
                                           self.clip_max_item).expand_as(item_weight))

                pro_weight = self.model.ebd_prototype.weight.data
                pro_weight.div_(torch.max(torch.norm(pro_weight, 2, 1, True),
                                           self.clip_max_pro).expand_as(pro_weight))  
                
                batch_data = [bd.cuda() for bd in batch_data]
                
                optimizer.zero_grad() 
                
                # Loss computation
                users, positems, negitems = batch_data

                c_posdist, i_posdist = self.model([users, positems])
                c_negdist, i_negdist = self.model([users, negitems])

                zero = torch.FloatTensor([0]).cuda()
                first_term = torch.max(c_posdist - c_negdist + opt.margin, zero)

                pp = [ipdict[it] for it in positems.tolist()]
                pn = [ipdict[it] for it in negitems.tolist()]

                pp = Variable(torch.FloatTensor(pp)).cuda()
                pn = Variable(torch.FloatTensor(pn)).cuda()

                second_term = torch.pow((i_posdist - i_negdist) - (pn - pp), 2)

                loss = first_term + opt.lamb * second_term

                loss = torch.mean(loss) 

                loss.backward()
                
                optimizer.step()
    
                batch_loss += loss.data.item()

            elapsed = time.time() - st
            evalt = time.time()
            
            with torch.no_grad():
                topHits, topNdcgs  = cal_measures(self.vld_loader, self.model, opt, ipdict)

                if (topHits[10] + topNdcgs[10])/2 > best_score:
                    best_score = (topHits[10] + topNdcgs[10])/2
                    
                    best_topHits = topHits
                    best_topNdcgs = topNdcgs
                    
                    c = 0
                    
                    test_topHits, test_topNdcgs = cal_measures(
                                    self.tst_loader, self.model, opt, ipdict)
                    
                evalt = time.time() - evalt 
            
            print(('(%.1fs, %.1fs)\tEpoch [%d/%d], TRN_ERR : %.4f, v_score : %5.4f, tHR@10 : %5.4f'% (elapsed, evalt, epoch, self.opt.num_epoch, batch_loss/len(self.trn_loader), (topHits[10] + topNdcgs[10])/2,  test_topHits[10])))

            batch_loss = 0
            
            c += 1
            if c > 5: break # Early-stopping
        
        print(('\nValid score@10 : %5.4f, HR@10 : %5.4f, NDCG@10 : %5.4f\n'% (((best_topHits[10] + best_topNdcgs[10])/2), best_topHits[10],  best_topNdcgs[10])))
        
        return test_topHits,  test_topNdcgs
            
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('\nn_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
        print('')

    def run(self, repeats):
        results = []
        rndseed = [19427, 78036, 37498, 87299, 60330] # randomly-generated seeds
        for i in range(repeats):
            print('\n💫 repeat: {}/{}'.format(i+1, repeats))
            random.seed(rndseed[i]); np.random.seed(rndseed[i]); torch.manual_seed(rndseed[i])
            self._reset_params()
            
            results.append(ins.train())
        
        results = np.array(results)
        
        hrs_mean = np.array([list(i.values()) for i in results[:,0]]).mean(0)
        ndcg_mean = np.array([list(i.values()) for i in results[:,1]]).mean(0)
        
        hrs_std = np.array([list(i.values()) for i in results[:,0]]).mean(0)
        ndcg_std = np.array([list(i.values()) for i in results[:,1]]).mean(0)
        
    
        print('*TST Performance\tTop2\tTop5\t\tTop10\t\tTop20\t')
        print('*HR means: {}'.format(', '.join(hrs_mean.astype(str))))
        print('*NDCG means: {}'.format(', '.join(ndcg_mean.astype(str))))
        
    def _reset_params(self):
        self.model = self.opt.model_class(self.opt).cuda()
```

```python id="IavTx4rs5Dnc" executionInfo={"status": "ok", "timestamp": 1628159887284, "user_tz": -330, "elapsed": 800, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
```

```python colab={"base_uri": "https://localhost:8080/"} id="-YA5Bpbi5HGT" executionInfo={"status": "ok", "timestamp": 1628159974053, "user_tz": -330, "elapsed": 621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3e3d0627-c88e-455a-c66b-cf901a8321a3"
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='cris', type=str)
parser.add_argument('--dataset', default='amazon_tools', type=str)    
parser.add_argument('--num_epoch', default=50, type=int)
parser.add_argument('--learning_rate', default=1e-2, type=float)    
parser.add_argument('--batch_size', default=4096, type=int)    
parser.add_argument('--margin', default=0.6, type=float)
parser.add_argument('--lamb', default=0.2, type=float)
parser.add_argument('--gamma', default=1.6, type=float)
parser.add_argument('--K', default=50, type=int)      
parser.add_argument('--numneg', default=10, type=int)
parser.add_argument('--gpu', default=0, type=int)

opt = parser.parse_args(args={})
opt
```

```python colab={"base_uri": "https://localhost:8080/"} id="5usPJsbt3giA" executionInfo={"status": "ok", "timestamp": 1628161122562, "user_tz": -330, "elapsed": 1137408, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8db241b5-ac70-4932-e173-edef6c487174"
torch.cuda.set_device(opt.gpu)

model_classes = {        
    'cris':CRIS,        
}  

dataset_path = './data/gold/{}'.format(opt.dataset)

opt.model_class = model_classes[opt.model_name]
opt.dataset_path = dataset_path

ins = Instructor(opt)

ins.run(5) 
```

```python id="h4eI7xTf97fm" executionInfo={"status": "ok", "timestamp": 1628161407705, "user_tz": -330, "elapsed": 875, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# save the model
torch.save(ins.model.state_dict(), './model/cris_v1.pt')
```

```python colab={"base_uri": "https://localhost:8080/"} id="cnlzrnrn-3Pp" executionInfo={"status": "ok", "timestamp": 1628161434314, "user_tz": -330, "elapsed": 964, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d605875d-ff58-41e4-d111-21278176a77d"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="4YvCdWpp-9M0" executionInfo={"status": "ok", "timestamp": 1628161502228, "user_tz": -330, "elapsed": 6219, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="14e77a9e-f684-4bbf-fd0a-b824a228722b"
!git add . && git commit -m 'ADD cris trained model v1, and official paper pdf' && git push origin main
```
