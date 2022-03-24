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

```python id="UV_mis-jdwLd" executionInfo={"status": "ok", "timestamp": 1629018114715, "user_tz": -330, "elapsed": 599, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="KRGLEjqMd3dV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629018120894, "user_tz": -330, "elapsed": 4421, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bd286723-c633-435e-cfa3-ecde24859a74"
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

```python id="Aa6AQmftAovn"
!git status
```

```python id="aG5PN_2EAovn"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="yyzDuLFlCr5p" -->
---
<!-- #endregion -->

```python id="3TIdoPoWCzRb" executionInfo={"status": "ok", "timestamp": 1629018124772, "user_tz": -330, "elapsed": 3885, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
import torch
import torch.nn as nn # for neural etworks
import torch.nn.parallel # for parallel computing
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
```

```python id="fuFx1NmvC8q0" executionInfo={"status": "ok", "timestamp": 1629018253598, "user_tz": -330, "elapsed": 603, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#u1.base training set # u1.test testing set using above data
training_set = pd.read_csv('./data/bronze/u1.base', delimiter='\t') #same like ratings data
test_set = pd.read_csv('./data/bronze/u1.test', delimiter='\t')
```

```python id="wm7dV92qDeL7" executionInfo={"status": "ok", "timestamp": 1629018681202, "user_tz": -330, "elapsed": 822, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
# creating two matrices for test and train in this matrices
# m(u,i) = will be the rating given by user 'u' for the movie 'i'.
# Observation in row and features in coloumns
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
  #why not 2d array instead of listlist : we are going to use pytorch
  #we will put the zero when user doe not rate the movie
  #so both the train and test matrics of 943*1682 will be of same size]
  new_data = []
  for i in range(1, 1+nb_users):
    id_movies = data[:,1][(data[:,0]==i)] #taking all the movie ids for which user has given the rating
    id_ratings = data[:,2][(data[:,0]==i)]
    ratings = np.zeros(nb_movies)
    ratings[id_movies-1] = id_ratings            # movie id starts for 1 therefore
    new_data.append(list(ratings))
  return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
# rows are obervations and columns are feature
# numpy array vs pytorch tensors vs tensorflow tensors
training_set = torch.FloatTensor(training_set)# the FLoatTensor class expects the list of list
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
# here we are just going to predict if only users like the movei or not
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0   # 1 and 2 are the movies which users did not like
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0   # 1 and 2 are the movies which users did not like
test_set[test_set >= 3] = 1
```

<!-- #region id="obmWySJmE8qy" -->
### Creating the architecture of the Neural Network
Probablisitc graphical model Bernouli RBM

Bernouli Sample creating a class for Restricted Boltzman Machine
<!-- #endregion -->

```python id="J1p9NUcrE5he" executionInfo={"status": "ok", "timestamp": 1629018683241, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class RBM():
  def __init__(self, nv, nh ):  # prbabilites of visible node given the hidden node #self is object o this class it self #nv no. of visible node
    self.W = torch.randn(nh,nv)   
    self.a = torch.randn(1,nh) # first dimention corresponding to batch #probability of the hidden node given the visible and vice ersa                                              
    self.b = torch.randn(1,nv)

  def sample_h(self, x): #x visible node
    """function to sample the hidden node
    this will return the values of hidden neuron given the values of ratings"""
    wx = torch.mm(x, self.W.t())
    activation = wx + self.a.expand_as(wx)
    p_h_given_v  = torch.sigmoid(activation)
    return p_h_given_v, torch.bernoulli(p_h_given_v)

  def sample_v(self,y): # x is hidden node
    wy = torch.mm(y, self.W) 
    activation = wy + self.b.expand_as(wy)
    p_v_given_h  = torch.sigmoid(activation)
    return p_v_given_h, torch.bernoulli(p_v_given_h)

  def train(self, v0, vk, ph0, phk):
    """v0 firts row, vk visible nodes after k
    ph0 prob od hidden nodes initiallly
    gibbs sampling"""
    #self.W += torch.mm( v0.t(), ph0 )   -  torch.mm(vk.t(), phk)
    self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
    self.b += torch.sum((v0-vk), 0)
    self.a += torch.sum((ph0-phk), 0)
```

```python id="cC2TfXGME7L_" executionInfo={"status": "ok", "timestamp": 1629018684705, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
nv = len(training_set[0])
nh = 100
batch_size = 100  # 1 for online learning
rbm = RBM(nv, nh)
```

<!-- #region id="TlmCFWkzE2Kl" -->
### Training the RBM
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="h0Af4NCME0_q" executionInfo={"status": "ok", "timestamp": 1629018698770, "user_tz": -330, "elapsed": 10888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8da1473d-ab3d-497f-9a75-838f67f8351f"
nb_epoch = 10
for epoch in range(1, nb_epoch+1):
  # we ned loss functron to check the differnce between predicted and real test ratigns
  train_loss = 0 
  s = 0.0
  for id_user in range(0, nb_users - batch_size, batch_size):
    vk = training_set[id_user:id_user+batch_size]
    v0 = training_set[id_user:id_user+batch_size]
    ph0,_ = rbm.sample_h(v0)
    for k in range(10):
       # markov chain monte carlo technique gibs sampling # random walk
      _,hk = rbm.sample_h(vk)
      _,vk = rbm.sample_v(hk)
      #lets update the weigth and bias
      vk[v0<0] = v0[v0<0]
    phk,_ = rbm.sample_h(vk)
    rbm.train(v0, vk, ph0, phk)
    train_loss += torch.mean(torch.abs(v0[v0>=0] -vk[v0>=0]))
    s+=1
  print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))
```

<!-- #region id="Kk5xbX7zFOCV" -->
### Testing the RBM
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wDHHw_WoFIaO" executionInfo={"status": "ok", "timestamp": 1629018715001, "user_tz": -330, "elapsed": 723, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ec6d10e8-056c-495c-ec0f-741469854ec3"
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]  # contains the real ratings
    if len(vt[vt>=0]) > 0:
       # markov chain monte carlo technique gibs sampling # random walk #blind walk
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))
```
