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

```python id="UV_mis-jdwLd" executionInfo={"status": "ok", "timestamp": 1629018862897, "user_tz": -330, "elapsed": 739, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="KRGLEjqMd3dV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629018866128, "user_tz": -330, "elapsed": 3255, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="36a7d6e5-fa2f-4fc2-898d-f7f9cf1a501e"
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

```python id="3TIdoPoWCzRb" executionInfo={"status": "ok", "timestamp": 1629019428558, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable #fro stochastic gradient decent
from torchsummary import summary
from tqdm import tqdm  # tqdm is for progress bar
import matplotlib.pyplot as plt
```

```python id="fuFx1NmvC8q0" executionInfo={"status": "ok", "timestamp": 1629018869912, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#u1.base training set # u1.test testing set using above data
training_set = pd.read_csv('./data/bronze/u1.base', delimiter='\t') #same like ratings data
test_set = pd.read_csv('./data/bronze/u1.test', delimiter='\t')
```

```python id="wm7dV92qDeL7" executionInfo={"status": "ok", "timestamp": 1629019012423, "user_tz": -330, "elapsed": 861, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
# creating two matrices for test and train in this matrices
# m(u,i) = will be the rating given by user 'u' for the movie 'i'.
# Observation in row and features in coloumns
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns
# Creating a matrix where cell u,i is a rating given by user u to the ith 
# movie and if no rating is given put 0 there
data = np.zeros((nb_users, nb_movies))
for i in range(len(training_set[:,0])):
  user = training_set[i,0]
  movie = training_set[i,1]
  data[user-1, movie-1]  = training_set[i,2]

#defining a function out of above for loop
def con(data):
  new_data = np.zeros((nb_users, nb_movies))
  for i in range(len(data[:,0])):
    user = data[i,0]
    movie = data[i,1]
    new_data[user-1, movie-1] = data[i,2]
  return new_data

new_data = con(training_set)
t = np.random.rand(3,3)
training_set = con(training_set)
test_set  = con(test_set)

# Converting the data into Torch tensors
training_set = torch.from_numpy(training_set)
test_set = torch.from_numpy(test_set)
training_set  = training_set.float()
test_set  = test_set.float()
```

<!-- #region id="obmWySJmE8qy" -->
### Creating the architecture of the Neural Network

Build a class stacked autoencoder, use the inheritance

we will create a child class form parant class Module and we will be able to use all the funtions and variable from parent class
<!-- #endregion -->

```python id="J1p9NUcrE5he" executionInfo={"status": "ok", "timestamp": 1629019076743, "user_tz": -330, "elapsed": 474, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class SAE(nn.Module):
  # in heritance is done
  def __init__(self, ):
    super(SAE, self).__init__()
    #we optimize it helps using functionalitiers form parent class
    #to get inherit methods from Modules
    #connection bwtween input and first encoded vector full connection
    self.fc1 = nn.Linear(nb_movies, 20)
    self.fc2 = nn.Linear(20,10)
    self.fc3 = nn.Linear(10,20)
    self.fc4 = nn.Linear(20,nb_movies)
    #specify an activation fucntions
    self.Activation = nn.Sigmoid()

  def forward(self,x):
    x = self.Activation(self.fc1(x))
    x = self.Activation(self.fc2(x))
    x = self.Activation(self.fc3(x))
    x = self.fc4(x) # we dont use activations here
    return x #this is the vector of predicted ratings
```

```python id="cC2TfXGME7L_" executionInfo={"status": "ok", "timestamp": 1629019102183, "user_tz": -330, "elapsed": 682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
sae=SAE()
criterion = nn.MSELoss()
# weight_decayas regularizes the learning rate after each epoch
opt = optim.RMSprop( params = sae.parameters(), lr = 0.01, weight_decay = 0.5)
```

<!-- #region id="TlmCFWkzE2Kl" -->
### Training the SAE

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="h0Af4NCME0_q" executionInfo={"status": "ok", "timestamp": 1629019426826, "user_tz": -330, "elapsed": 233298, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e0efeba7-6b6a-49c4-8085-0d585ecfe4e9"
#optimize the code for bigger dataset
epochs = 200
train_loss_list = []
for epoch in range(1, epochs+ 1):
  train_loss = 0
  #excluding the users who did not rate
  s = 0.0 #float
  for id_user in range(nb_users):
    input = training_set[id_user]
    input = Variable(input).unsqueeze(0)#create a new dimenion at 0 index in shape
    target = input.clone()
    #output = sae(input)#additional dimenstion for the batch
    if (torch.sum(input)>0):
       output = sae(input)
       #optimizing memory and computation
       target.require_grad =  False
       output[target == 0] = 0
       loss = criterion(output, target)
       mean_corrector = nb_movies/(float(torch.sum(target.data>0)) + 1e-10)
       loss.backward()#weights will increase or decrease
       train_loss = train_loss + torch.sqrt((loss*mean_corrector))
       s = s + 1.
       opt.step()# by how weight will decrease or increase
  if epoch%20==0:  
    print('Epoch: '+ str(epoch) + " loss : " + str(train_loss/s) )
  train_loss_list.append(train_loss)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="thm1AzezHHWH" executionInfo={"status": "ok", "timestamp": 1629019430278, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c1c0ab6f-0f09-42b2-f51f-c648e08163c1"
x = np.array(list(range(1,201, 1)))
plt.plot(x, np.array(train_loss_list))
plt.title(" Training Loss ")
plt.xlabel(" No. of epochs ")
plt.ylabel(" Loss ")
plt.show()
```

<!-- #region id="Kk5xbX7zFOCV" -->
### Testing the SAE
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wDHHw_WoFIaO" executionInfo={"status": "ok", "timestamp": 1629019430967, "user_tz": -330, "elapsed": 704, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1411e120-581f-48f8-f084-00819e26ca3c"
epochs = 200
train_loss_list = []

test_loss = 0
#excluding the users who did not rate
s = 0.0 #float
for id_user in range(nb_users):
  input = training_set[id_user]
  #here also we will keep it as training_set only as we want to predict the ratings for the movies which user has not watched yet
  #then we will compare this with the data in the test  set as test set has the ratings which are not there in the training set
  input = Variable(input).unsqueeze(0)#create a new dimenion at 0 index in shape
  target = Variable(test_set[id_user].unsqueeze(0))
  #output = sae(input)#additional dimenstion for the batch
  if (torch.sum(input)>0):
      output = sae(input) #these are rating for other movies
      #optimizing memory and computation
      target.require_grad =  False
      output[target == 0] = 0
      loss = criterion(output, target)
      mean_corrector = nb_movies/(float(torch.sum(target.data>0)) + 1e-10)
      test_loss = test_loss + torch.sqrt((loss*mean_corrector))
      s = s + 1.
print(" Test Loss : " + str(test_loss/s) )
```
