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

```python id="Dg8frDmMWhHA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628159068629, "user_tz": -330, "elapsed": 5707, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c0008e19-98c1-4848-cc33-6a400ac7d8f3"
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

```python id="ziquRWpm0Q48" executionInfo={"status": "ok", "timestamp": 1628159068630, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0, './code')
```

```python id="tA3NqnRH0FNT" executionInfo={"status": "ok", "timestamp": 1628159072141, "user_tz": -330, "elapsed": 3516, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import pdb
import time
import pdb
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from torch.autograd import Variable
from sklearn.metrics import f1_score

from dataloader_interest import DataLoader
```

```python colab={"base_uri": "https://localhost:8080/"} id="KsDaIYgj04Af" executionInfo={"status": "ok", "timestamp": 1628159073036, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3743dc54-bae6-48bf-fbaa-3d72385ce3cd"
torch.set_num_threads(4)

random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
```

```python id="YJCmsvA10CyS" executionInfo={"status": "ok", "timestamp": 1628159073036, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class INTEREST_LEARNER(nn.Module):
    def __init__(self, opt):
        super(INTEREST_LEARNER, self).__init__()
        
        self.hd = opt.hidden_dim        
        
        self.proj = nn.Linear(self.hd*2,1)
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hd,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
    
    def forward(self, batch_data):
        frequency_bins = batch_data[1].float().cuda()
        
        output, _ = self.lstm(frequency_bins.unsqueeze(-1))
        interim = output[:,-1,:]
    
        prob = self.proj(interim)
        
        return nn.Sigmoid()(prob)  
```

```python id="cSKm6WwM0-o-" executionInfo={"status": "ok", "timestamp": 1628159073037, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class Instructor:
    def __init__(self, opt):
        self.opt = opt
        
        self.data_loader = DataLoader(self.opt)
        self.trn_loader, self.vld_loader, self.tst_loader = self.data_loader.get_loaders()

        trnlen = self.trn_loader.dataset.timediffs.shape[1]        

        print('TRN labels: {}'.format(Counter(self.trn_loader.dataset.labels)))
        print('VLD labels: {}'.format(Counter(self.vld_loader.dataset.labels)))
        print('TST labels: {}'.format(Counter(self.tst_loader.dataset.labels)))
        
        self.model = self.opt.model_class(self.opt).cuda()
        
        self._print_args()                
        
        
    def train(self):
        criterion = nn.BCELoss(reduction='none')
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.learning_rate)        
        
        best_vf1 = 0
        vld_f1 = 0
        tst_f1 = 0        
        batch_loss = 0        
        
        for epoch in range(self.opt.num_epoch):                        
            
            st = time.time()
            trn_outputs, trn_labels = [], []            
            for i, batch_data in enumerate(self.trn_loader):            
                batch_only_data = batch_data[:-1] # cuda will be called in the model
                labels = batch_data[-1].float().cuda()    
                
                if (labels>1).sum() != 0:
                    print('Label list contains an element not 0 or 1')
                    pdb.set_trace()
                
                class_weight = (labels == 1).float()                
                class_weight *= opt.pos_weight
                class_weight[class_weight==0] = (1-opt.pos_weight)
                class_weight = nn.functional.softmax(class_weight, dim=0)
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_only_data).view(-1)
                loss = criterion(outputs, labels)
                
                loss = (loss * class_weight).sum()
                
                loss.backward()
                
                optimizer.step()

                batch_loss += loss.data.item()
                
                trn_outputs.append(outputs)
                trn_labels.append(labels)
        
            elapsed = time.time() - st

            evalt = time.time()
            
            trn_outputs = (torch.cat(trn_outputs) >= 0.5).float()    
            trn_labels = torch.cat(trn_labels)    
                        
            trn_f1 = f1_score(trn_labels.cpu().numpy(), trn_outputs.cpu().numpy(), average='binary')            
                        
            # Evaluation
            with torch.no_grad():
                
                vld_iids, vld_outputs, vld_labels = [], [], []
                for i, batch_data in enumerate(self.vld_loader):
                    batch_only_data = batch_data[:-1] # cuda will be called in models
                    
                    vld_iids += list(batch_data[0])                    
                    labels = batch_data[-1].float().cuda()

                    outputs = self.model(batch_only_data).view(-1)

                    vld_outputs.append(outputs)
                    vld_labels.append(labels)
                    
                vld_probs = torch.cat(vld_outputs)
                vld_outputs = (vld_probs >= 0.5).float()   
                vld_labels = torch.cat(vld_labels)    
                
                
                vld_f1 = f1_score(vld_labels.cpu().numpy(), vld_outputs.cpu().numpy(), average='binary')
                
    
                if vld_f1 > best_vf1:
                    best_vf1 = vld_f1

                    # Save ISSs of items
                    item_interest = np.vstack([np.array(vld_iids), vld_probs.cpu().numpy()])
                    recpath = '/'.join(opt.dataset_path.split('/')[:-1])+'/'
                    if not os.path.exists(recpath): os.makedirs(recpath)
                    np.save(open(recpath+'/interest_prob', 'wb'), item_interest)
                    
                    tst_outputs, tst_labels = [], []
                    for k, batch_data in enumerate(self.tst_loader):
                        batch_only_data = batch_data[:-1]
                        labels = batch_data[-1].float().cuda()

                        outputs = self.model(batch_only_data).view(-1)

                        tst_outputs.append(outputs)
                        tst_labels.append(labels)

                    tst_outputs = (torch.cat(tst_outputs) >= 0.5).float()   
                    tst_labels = torch.cat(tst_labels)    

                    tst_f1 = f1_score(tst_labels.cpu().numpy(), tst_outputs.cpu().numpy(), average='binary')
                    
            evalt = time.time() - evalt
                    
            print(('(%.1fs, %.1fs)\tEpoch [%d/%d], trn_e : %5.4f, trn_f1 : %4.3f, vld_f1 : %4.3f, tst_f1 : %4.3f'% (elapsed, evalt, epoch, self.opt.num_epoch, batch_loss/len(self.trn_loader), trn_f1, vld_f1,  tst_f1)))            
            
            batch_loss =0
                    
        print('VLD F1 and TST:\t{}\t{}'.format(best_vf1, tst_f1))
        
    
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
```

```python id="sGHqW6kf1BEo" executionInfo={"status": "ok", "timestamp": 1628159073038, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
```

```python colab={"base_uri": "https://localhost:8080/"} id="GfUcZL8f1G7q" executionInfo={"status": "ok", "timestamp": 1628159073038, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2ee4d173-c018-4d98-8fab-735926acb717"
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='intlearn', type=str)
parser.add_argument('--dataset', default='amazon_tools', type=str)    
parser.add_argument('--period', default=16, type=float)
parser.add_argument('--binsize', default=8, type=int)
parser.add_argument('--learning_rate', default=1e-2, type=float)
parser.add_argument('--l2reg', default=1e-4, type=float)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)    
parser.add_argument('--hidden_dim', default=64, type=int)    
parser.add_argument('--pos_weight', default=1e-2, type=float)   
parser.add_argument('--gpu', default=0, type=int)       

opt = parser.parse_args(args={})
opt
```

```python colab={"base_uri": "https://localhost:8080/"} id="wOA1pP3r0y7b" executionInfo={"status": "ok", "timestamp": 1628159197547, "user_tz": -330, "elapsed": 124517, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1789a85d-f997-4c6b-9970-c8470dd0e4b7"
torch.cuda.set_device(opt.gpu)

model_classes = {
    'intlearn': INTEREST_LEARNER,      
}
    
dataset_path = './data/silver/{}'.format(opt.dataset)    

opt.model_class = model_classes[opt.model_name]
opt.dataset_path = dataset_path

ins = Instructor(opt)
ins.train()
```

```python colab={"base_uri": "https://localhost:8080/"} id="uWF9XsL21h6P" executionInfo={"status": "ok", "timestamp": 1628159209088, "user_tz": -330, "elapsed": 507, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c0cc185c-01d6-45db-e1ef-1c0406d9145f"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="lDWMsGJ02eVL" executionInfo={"status": "ok", "timestamp": 1628159354722, "user_tz": -330, "elapsed": 832, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a007a1a1-f997-482a-a482-2dad957641e2"
!git add . && git commit -m 'ADD interest model training probabilities in silver layer' && git push origin main
```
