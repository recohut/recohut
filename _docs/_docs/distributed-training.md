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

```python id="V9CYiRDI6Gmm"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import time
import argparse


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64) # 4608 is basically 12 X 12 X 32
        self.fc2 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        return op
    
    
def train(args):
    torch.manual_seed(0)
    device = torch.device("cpu")
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1302,), (0.3069,))])),
        batch_size=128, shuffle=True)  
    model = ConvNet()
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    model.train()
    for epoch in range(args.epochs):
        for b_i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            loss = F.nll_loss(pred_prob, y) # nll is the negative likelihood loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if b_i % 10 == 0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i, len(train_dataloader),
                    100. * b_i / len(train_dataloader), loss.item()))
```

```python id="SCzqTz_x6NAr"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int)
    args = parser.parse_args(args={})
    start = time.time()
    train(args)
    print(f"Finished training in {time.time()-start} secs")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["dbf0b970c11548839167659a424740ae", "5eb82411df874c19b09f51e8f1f5cf1f", "d02ab9fc49744aef8b42bbba32788b2f", "ea0738c7aa964d41871501439e0ca29a", "3fcf70247adc4c6a8eeffdf308cde9a0", "c7ce1753f7434cc3be8e4e52f33e7c5d", "31719fd810ba419ebc8d2b074ab4d5a9", "a9221a3c9c0348a29fff35b0c16d61d8", "d99ff06e7152497d9dfe3fae2c7784bf", "0796b0c926e74b11a9bfb8b3da5d13db", "b323f44b32514b5eb53afe0fb4f04c00", "3d46665df63b4790975a3301d84d3f7c", "d3df1935e8e94019818ae517d07385c2", "69177f07295b4166adc252170e75551e", "c58302fc8bbb4953b90a699ec3ba073f", "5a570fdc00784c4a852b12da9b0b96c2", "4a247bfe2fdd4d6cb6f3aba7e51c8950", "e6c1453b792a453a803ee89d5c7294fe", "b1a5de08e6464c0bb91287f93a964e28", "b4da523784804e69be1b1208b748052f", "0f5f910c828c4e37abb076cac98cb4c4", "a37530f4f6e14a6a937c1e3ce43e33d2", "b5b443c677134b7389a95e5fc63421d8", "7511d4e422d74874a5f35380e66ef69d", "1e1b942b267b4176b14464bd71a61e91", "d28c890af7ee4f9187e51561d42c3507", "978a484e0b6f402c98f6ee5fd66fc079", "72e68755b04f43be8c29a5d4286bd07b", "4fbc9300d5b4451681e1fe67f21accf7", "52fbc5d9d0354f7bbd2f832cf6a925d2", "55358baa143140c8b939561c9947f6ae", "bbae73eeb20346c19a7a8d10173891bd", "a52ad4f8255649728de4012d77bd3904", "d55421641bab4770b64d653eec59ff3a", "d49834477ff44bbfaaceb0c471e880dd", "71cea5e01cd94038b1dd5a933b25a453", "f3b199a50381443a9c922346b8524a99", "6cfb854fb17b4a399ff4bcc369a2d3fc", "52c55436b597486fa7878b62a4a50069", "f2451791d1394f05817ff70f965b7333", "6d989e0e9d0645178e185187c3131192", "325a3c68d7464ce78e76d8d715d32b64", "1c3ec0bea9a94200bb7db20af34f1cd1", "d740336ec9d947258e1866b7bf3282bf"]} id="uSowCnnr6PZO" executionInfo={"status": "ok", "timestamp": 1631264040877, "user_tz": -330, "elapsed": 52320, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="25c18abe-0b7b-4e5a-f621-440e00225d2c"
main()
```

<!-- #region id="wjYsQDIz6RiF" -->
It took roughly 50 seconds to train for 1 epoch, which equates to 469 batches, each of which has 128 data points. The only exception is the last batch, which has 32 fewer data points than usual (as there are 60,000 data points in total).

At this point, it is important to know what kind of machine this model is being trained on so that we know the reference context
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="W4PtFvnS6foj" executionInfo={"status": "ok", "timestamp": 1631264123379, "user_tz": -330, "elapsed": 1762, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7ea86811-4a6b-4aa3-f952-65c1f6b9dca2"
# !sudo apt-get install inxi
!inxi -F
```

<!-- #region id="D_Teewwj8YNf" -->
While torch.multiprocessing helps spawn multiple Python processes within a machine (typically, we may spawn as many processes as there are CPU cores in the machine), torch.distributed enables communications between different machines as they work together to train the model. During execution, we need to explicitly launch our model training script from within each of these machines.

One of the built-in PyTorch communication backends, such as Gloo, will then take care of the communication between these machines. Inside each machine, multiprocessing will take care of parallelizing the training task across several processes.
<!-- #endregion -->

```python id="hVdOId9c6x_E"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.multiprocessing as mp
import torch.distributed as dist

import os
import time
import argparse


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64) # 4608 is basically 12 X 12 X 32
        self.fc2 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        return op
     

def train(cpu_num, args):
    rank = args.machine_id * args.num_processes + cpu_num                        
    dist.init_process_group(                                   
    backend='gloo',                                         
    init_method='env://',                                   
    world_size=args.world_size,                              
    rank=rank                                               
    ) 
    torch.manual_seed(0)
    device = torch.device("cpu")
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1302,), (0.3069,))]))  
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    train_dataloader = torch.utils.data.DataLoader(
       dataset=train_dataset,
       batch_size=args.batch_size,
       shuffle=False,            
       num_workers=0,
       sampler=train_sampler)
    model = ConvNet()
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    model = nn.parallel.DistributedDataParallel(model)
    model.train()
    for epoch in range(args.epochs):
        for b_i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            loss = F.nll_loss(pred_prob, y) # nll is the negative likelihood loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if b_i % 10 == 0 and cpu_num==0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i, len(train_dataloader),
                    100. * b_i / len(train_dataloader), loss.item()))
         
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-machines', default=1, type=int,)
    parser.add_argument('--num-processes', default=1, type=int)
    parser.add_argument('--machine-id', default=0, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args(args={})
    
    args.world_size = args.num_processes * args.num_machines                
    os.environ['MASTER_ADDR'] = '127.0.0.1'              
    os.environ['MASTER_PORT'] = '8892'      
    start = time.time()
    mp.spawn(train, nprocs=args.num_processes, args=(args,))
    print(f"Finished training in {time.time()-start} secs")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 612} id="F2-NQ6BT7u2V" executionInfo={"status": "error", "timestamp": 1631264381729, "user_tz": -330, "elapsed": 1473, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="567853d3-763f-40fa-a64a-d949187e7230"
main()
```

```python id="mlZusrRX7vUv"

```
