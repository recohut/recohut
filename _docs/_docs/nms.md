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

<!-- #region id="M2GX2zB_Pl9h" -->
# Neural Memory Streaming Recommender Networks with Adversarial Training
<!-- #endregion -->

<!-- #region id="TuygM6y1WO66" -->
NMRN is a streaming recommender model based on neural memory networks with external memories to capture and store both long-term stable interests and short-term dynamic interests in a unified way. An adaptive negative sampling framework based on Generative Adversarial Nets (GAN) is developed to optimize our proposed streaming recommender model, which effectively overcomes the limitations of classical negative sampling approaches and improves the effectiveness of the model parameter inference.
<!-- #endregion -->

<!-- #region id="HPDRZQSvZzNs" -->
![](https://github.com/recohut/incremental-learning/raw/a6fdcde2e8af7ebfd9f5efd278c487e0e9560cb3/docs/_images/T063788_1.png)
<!-- #endregion -->

<!-- #region id="kw5elHdCWrGo" -->
Specifically, the external memories of NMRN compose the key memory and the value memory. Given a new user-item interaction pair(u , v) arriving at the system in real time, NMRN first generates a soft address from the key memory, activated by u. The addressing process is inspired by recent advances in attention mechanism. The attention mechanism applied in recommender systems is useful to improve the retrieval accuracy and model interpretability. The fundamental idea of this attention design is to learn a weighted representation across the key memory, which is converted into a probability distribution by the Softmax function as the soft address.Then, NMRN reads from the value memory based on the soft address, resulting in a vector that represents both long-term stable and short-term emerging interests of user u. Inspired by the success of pairwise personalized ranking models (e.g., BPR) in top-k recommendations, we adopt the Hinge-loss in our model optimization. As the number of unobserved examples is very huge, we use the negative sampling method to improve the training efficiency.

Most existing negative sampling approaches use either random sampling or popularity-biased sampling strategies. However, the majority of negative examples generated in these sampling strategies can be easily discriminated from observed examples, and will contribute little towards the training, because sampled items could be completely unrelated to the target user. Besides, these sampling approaches are not adaptive enough to generate adversarial negative examples, because (1) they are static and thus do not consider that the estimated similarity or proximity between a user and an item changes during the learning process. For example, the similarity between user u and a sampled noise item v is high at the beginning, but after several gradient descent steps it becomes low; and (2) these samplers are global and do not reflect how informative a noise item is w.r.t. a specific user.

In light of this, we use an adaptive noise sampler based on a Generative Adversarial Network (GAN) to optimize the model, which considers both the specific user and the current values of the model parameters to adaptively generate “difficult” and informative negative examples. Moreover, in order to simultaneously capture the first-order similarity between users and items as well as the second-order similarities between users and between items to learn robust representations of users and items, we use the Euclidean distance to measure the similarity between a user and an item instead of the widely adopted dot product.
<!-- #endregion -->

<!-- #region id="u6H8sp5JZ2YT" -->
![](https://github.com/recohut/incremental-learning/raw/a6fdcde2e8af7ebfd9f5efd278c487e0e9560cb3/docs/_images/T063788_2.png)
<!-- #endregion -->

<!-- #region id="uRxvbQ0NWunq" -->


The Architecture of the model. The total numbers of users and items are denoted as $N$ and $M$. We denote a user-item interaction pair at time $t$ as $(u_t , v_t)$ while $v_t^-$ is a sampled negative item for a specific user at time $t$.
<!-- #endregion -->

```python id="xf6MOr-0PnDH"
import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.parameter import Parameter
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from math import log
```

```python id="qgxBOo0ZVYWO"
#some parameters of this model
N, D_in, H = 64, 20, 10
safety_margin_size = 3
Max_sampleNum = 50
total_item_num = 500
```

```python id="sjSGYoItVata"
def euclidean_distance(a, b):
    return (a-b).pow(2).sum(1).sqrt()
```

```python id="gAFIiaF1VhM8"
value_memory = torch.randn(H, D_in, requires_grad=True)
key_memory = torch.randn(H, D_in, requires_grad=True)

x = torch.randn(500, D_in)
y = torch.randn(500, D_in)
```

```python id="8y1HvCc1Vi21"
class CandidateDataset(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.as_tensor(x,dtype=torch.float)
        self.y_data = torch.as_tensor(y,dtype=torch.float)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
```

```python id="DLJ51QrTVlin"
my_dataset = CandidateDataset(x, y)
train_loader = DataLoader(dataset = my_dataset, batch_size = 32, shuffle=True)
```

```python id="lg7CVMoJVmwz"
class generator(nn.Module):
    def __init__(self, D_in):
        super(generator, self).__init__()

        self.fc1 = nn.Linear(2*D_in, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)  # Prob of Left

    def forward(self, x, y):
        cat_xy = torch.cat([x,y],1)
        hidden1 = F.relu(self.fc1(cat_xy))
        hidden2 = F.relu(self.fc2(hidden1))
        output = torch.sigmoid(self.fc3(hidden2))
        return output
```

```python id="_1OQDusHVn9E"
class Discriminator(torch.nn.Module):
    def __init__(self, D_in, H, N, key_memory, value_memory):
        super(Discriminator, self).__init__()
        self.key_memory = Parameter(torch.Tensor(H, D_in))
        self.value_memory = Parameter(torch.Tensor(H, D_in))
        self.key_memory.data = key_memory
        self.value_memory.data = value_memory
        #self.value_memory.requires_grad=True
        self.fc_erase = nn.Linear(D_in, D_in)
        self.fc_update = nn.Linear(D_in, D_in)
        self.D_in = D_in
        self.N = N
        self.H = H
    def forward(self, user, item):
        output_list = []
        e_I = torch.ones(self.H, self.D_in)
        
        for i in range(self.N):
            #Memory Adderssing
            Attention_weight = torch.empty(self.H)

            for j in range(self.H):
                Attention_weight[j] = -euclidean_distance(user[i].unsqueeze(0), self.key_memory[j,:])


            #select value memory by attention

            Attention_weight = Attention_weight.softmax(0)
            s = Attention_weight.matmul(self.value_memory)

            output = euclidean_distance(s, item[i].unsqueeze(0))
            output_list.append(output)
            
            #update value memory by item vector
            e_t = self.fc_erase(item[i].unsqueeze(0)).sigmoid()
            self.value_memory.data = self.value_memory * (e_I - Attention_weight.unsqueeze(0).t().matmul(e_t))

            a_t = self.fc_update(item[i].unsqueeze(0)).tanh()
            self.value_memory.data = self.value_memory + Attention_weight.unsqueeze(0).t().matmul(a_t)
        
        return output_list
```

```python id="lldToo4eVpv9"
def ngtv_spl_by_unifm_dist(user):
    ngtv_item = torch.randn(1, D_in)
    return ngtv_item

def ngtv_spl_by_generator(user):
    ngtv_item_id = list(torch.multinomial(prob_ngtv_item, 1, replacement=True))[0]
    ngtv_item = dict_item_id2vec[candidate_ngtv_item[ngtv_item_id].item()]
    return ngtv_item_id, ngtv_item

def G_pretrain_per_datapoint(user, generator, model):
    ngtv_item_id, ngtv_item = generator(user)
    neg_rslt = model(user.unsqueeze(0), ngtv_item)[0]
    return -neg_rslt, ngtv_item_id

def D_pretrain_per_datapoint(user, item, generator):
    ps_rslt = model(user.unsqueeze(0),item.unsqueeze(0))[0]
    for i in range(Max_sampleNum):
        weight = log((total_item_num-1)//(i+1) + 1)
        neg_rslt = model(user.unsqueeze(0), generator(user))[0]
        loss = weighted_hinge_loss(ps_rslt, neg_rslt, safety_margin_size, weight)
        if(loss > 0):
            break;
    return loss
    
def weighted_hinge_loss(ps_rslt, neg_rslt, safety_margin_size, weight):
    return weight * F.relu(safety_margin_size + ps_rslt - neg_rslt)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1uksFi-TVvaw" executionInfo={"status": "ok", "timestamp": 1635230723030, "user_tz": -330, "elapsed": 95726, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d8566b76-ab5c-4012-c812-a83630cfc64b"
# pre-training of discriminator
model = Discriminator(20, 10, 1, key_memory, value_memory)
total_loss = []
optimizer = Adam(model.parameters(), lr= 0.01)

for epoch in range(50):
    epoch_loss = 0.
    for batch_id, (x, y) in enumerate(train_loader):
        losses = torch.empty(x.shape[0])
        for i in range(x.shape[0]):
            losses[i] = D_pretrain_per_datapoint(x[i],y[i], ngtv_spl_by_unifm_dist)

        mini_batch_loss = torch.sum(losses)
        epoch_loss += float(mini_batch_loss.data)
        
        model.zero_grad()
        mini_batch_loss.backward()
        optimizer.step()
        
    print('epoch_' + str(epoch) + ':  loss=' + str(epoch_loss))
    total_loss.append(epoch_loss)    
```

```python id="_LKhJLxWVzKl"
dict_user_id2vec = {}
dict_item_id2vec = {}
candidate_ngtv_item = torch.randn(100)
prob_ngtv_item = torch.randn(100)
```

```python id="aqOWX1MIWBlK"
for i in candidate_ngtv_item:
    dict_item_id2vec[i.item()] = torch.randn(1,20)
```

<!-- #region id="LHalxg4hWcGb" -->
## Training procedure
<!-- #endregion -->

<!-- #region id="uZhUQDQzZ457" -->
![](https://github.com/recohut/incremental-learning/raw/a6fdcde2e8af7ebfd9f5efd278c487e0e9560cb3/docs/_images/T063788_2.png)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Q20Gq0PqV_Jc" executionInfo={"status": "ok", "timestamp": 1635231474888, "user_tz": -330, "elapsed": 750287, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="75abddd1-c09d-45be-be92-51399bd8f7ec"
#pre-training of generator
gener = generator(20)
total_reward = []
total_loss = []
optimizer_G = Adam(gener.parameters(), lr= 0.01)

for epoch in range(50):
    epoch_reward = 0.
    epoch_loss = 0.
    for batch_id, (x, y) in enumerate(train_loader):
        reward = torch.empty(x.shape[0])
        mini_batch_loss = torch.empty(x.shape[0])
        
        for data_point in range(x.shape[0]):
            #state = x[i]
            #action_pool = []
            #reward_pool = []
            candidate_ngtv_item = torch.randn(100)
            prob_ngtv_item = torch.randn(100)
            for i in candidate_ngtv_item:
                dict_item_id2vec[i.item()] = torch.randn(1,20)
            
            for i in range(candidate_ngtv_item.shape[0]):
                prob_ngtv_item[i] = gener(x[data_point].unsqueeze(0), dict_item_id2vec[candidate_ngtv_item[i].item()])
            prob_ngtv_item = torch.softmax(prob_ngtv_item,0)
            reward[data_point], ngtv_item_id = G_pretrain_per_datapoint(x[data_point], ngtv_spl_by_generator, model)
            mini_batch_loss[data_point] = (-reward[data_point]) * torch.log(prob_ngtv_item[ngtv_item_id])
        rewards = torch.sum(reward)
        mini_batch_losses = torch.sum(mini_batch_loss)
        epoch_loss = epoch + float(mini_batch_losses.data)
        epoch_reward = epoch + float(rewards.data)
        
        gener.zero_grad()
        mini_batch_losses.backward()
        optimizer_G.step()
    
    print('epoch_' + str(epoch) + ':  loss=' + str(epoch_loss) + '  reward=  ' + str(epoch_reward))
    total_loss.append(epoch_loss)  
```

<!-- #region id="mjV33gv8WVJC" -->
## Links

[https://github.com/lzzscl/KBGAN-KV_MemNN/tree/master](https://github.com/lzzscl/KBGAN-KV_MemNN/tree/master/)

[Neural Memory Streaming Recommender Networks with Adversarial Training](https://dl.acm.org/doi/epdf/10.1145/3219819.3220004)
<!-- #endregion -->
