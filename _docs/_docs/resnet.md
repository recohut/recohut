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

<!-- #region id="jn814bda0Jc9" -->
# Resnet-18 Image Classifier on STL-10 dataset
<!-- #endregion -->

<!-- #region id="xlcJiz05lCqg" -->
<!-- #endregion -->

<!-- #region id="fKdB6UTWStSN" -->
## Loading and Processing Data 
<!-- #endregion -->

```python id="Qp5BM4bnStST" colab={"base_uri": "https://localhost:8080/", "height": 100, "referenced_widgets": ["2db40f1d829d47c8bc40b904570fa802", "63bb9f9d12c545f6aad9eb4744b9a9f1", "9fd2f3cb31bb4726b7045ddcdd820f82", "8953e4890a6e4084b39ec8f3280ed9e9", "25f7fb68ecc74f799ddb8c0b0a44b03e", "c6760e394df94a3a86fac9a2d281f049", "4a42688b18af4538ac8c483f948cd739", "219c83e18406426893d6abaaeea1c2ce"]} executionInfo={"status": "ok", "timestamp": 1608533035138, "user_tz": -330, "elapsed": 183224, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9f0e4f30-8f81-43c7-cb62-63373d433247"
from torchvision import datasets
import torchvision.transforms as transforms
import os

## path to store/load data
path2data="./data"
if not os.path.exists(path2data):
    os.mkdir(path2data)
    
## define transformation
data_transformer = transforms.Compose([transforms.ToTensor()])
    
## loading data
train_ds=datasets.STL10(path2data, split='train', download=True,transform=data_transformer)

## print out data shape
print(train_ds.data.shape)
```

```python id="afs4gp0bStS0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533040022, "user_tz": -330, "elapsed": 3268, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6128d1fa-498b-4086-c40e-b68369034758"
import collections

## get labels
y_train=[y for _,y in train_ds]

## count labels
counter_train=collections.Counter(y_train)
print(counter_train)
```

```python id="GFtinssgStTG" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533047256, "user_tz": -330, "elapsed": 10320, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3165d5a7-c032-4f16-de01-873c7a7e97d8"
## loading data
test0_ds=datasets.STL10(path2data, split='test', download=True,transform=data_transformer)
print(test0_ds.data.shape)
```

```python id="kGM_eDLsStTW" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533050558, "user_tz": -330, "elapsed": 13472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="08c3afa9-bcc0-4998-b9cd-d2ef63941b7c"
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

indices=list(range(len(test0_ds)))
y_test0=[y for _,y in test0_ds]
for test_index, val_index in sss.split(indices, y_test0):
    print("test:", test_index, "val:", val_index)
    print(len(val_index),len(test_index))
```

```python id="deBE4oF2StTj"
from torch.utils.data import Subset

val_ds=Subset(test0_ds,val_index)
test_ds=Subset(test0_ds,test_index)
```

```python id="J9Vw_zluStTr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533052906, "user_tz": -330, "elapsed": 15027, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a7a20e48-4d56-4227-d824-fbd540ca0f81"
import collections
import numpy as np

## get labels
y_test=[y for _,y in test_ds]
y_val=[y for _,y in val_ds]

counter_test=collections.Counter(y_test)
counter_val=collections.Counter(y_val)

print(counter_test)
print(counter_val)
```

```python id="wlAZX4uaStT0" colab={"base_uri": "https://localhost:8080/", "height": 240} executionInfo={"status": "ok", "timestamp": 1608533055545, "user_tz": -330, "elapsed": 17475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="258f242c-d63d-40f0-af6c-45aa5c6293f5"
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

## fix random seed
np.random.seed(0)

def show(img,y=None,color=True):
    npimg = img.numpy()
    npimg_tr=np.transpose(npimg, (1,2,0))
    plt.imshow(npimg_tr)
    if y is not None:
        plt.title("label: "+str(y))
        
grid_size=4
rnd_inds=np.random.randint(0,len(train_ds),grid_size)
print("image indices:",rnd_inds)

x_grid=[train_ds[i][0] for i in rnd_inds]
y_grid=[train_ds[i][1] for i in rnd_inds]

x_grid=utils.make_grid(x_grid, nrow=4, padding=2)
print(x_grid.shape)

## call helper function
plt.figure(figsize=(10,10))
show(x_grid,y_grid)
```

```python id="aw8C31sbStT_" colab={"base_uri": "https://localhost:8080/", "height": 240} executionInfo={"status": "ok", "timestamp": 1608533055546, "user_tz": -330, "elapsed": 17326, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="20b5247c-a999-4111-d6fa-1c3768086139"
np.random.seed(0)

grid_size=4
rnd_inds=np.random.randint(0,len(val_ds),grid_size)
print("image indices:",rnd_inds)

x_grid=[val_ds[i][0] for i in rnd_inds]
y_grid=[val_ds[i][1] for i in rnd_inds]

x_grid=utils.make_grid(x_grid, nrow=4, padding=2)
print(x_grid.shape)

## call helper function
plt.figure(figsize=(10,10))
show(x_grid,y_grid)
```

```python id="6bOwEaz0StUF" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533058383, "user_tz": -330, "elapsed": 20073, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c7eaa28f-1e81-44bc-b08b-a9457a961514"
import numpy as np

## RGB mean and std 
meanRGB=[np.mean(x.numpy(),axis=(1,2)) for x,_ in train_ds]
stdRGB=[np.std(x.numpy(),axis=(1,2)) for x,_ in train_ds]

meanR=np.mean([m[0] for m in meanRGB])
meanG=np.mean([m[1] for m in meanRGB])
meanB=np.mean([m[2] for m in meanRGB])

stdR=np.mean([s[0] for s in stdRGB])
stdG=np.mean([s[1] for s in stdRGB])
stdB=np.mean([s[2] for s in stdRGB])

print(meanR,meanG,meanB)
print(stdR,stdG,stdB)
```

```python id="SgjPzUQBStUN"
train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomVerticalFlip(p=0.5),  
    transforms.ToTensor(),
    transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])])
                 

test0_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
    ])   
```

```python id="Ga16zTVOStUW"
## overwrite the transform functions
train_ds.transform=train_transformer
test0_ds.transform=test0_transformer
```

```python id="0EvYJlv2StUd" colab={"base_uri": "https://localhost:8080/", "height": 257} executionInfo={"status": "ok", "timestamp": 1608533058391, "user_tz": -330, "elapsed": 19599, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="da5c4eea-aecd-48b9-aaa0-c105b32ea328"
import torch
np.random.seed(0)
torch.manual_seed(0)

## make a grid
grid_size=4
rnd_inds=np.random.randint(0,len(train_ds),grid_size)
print("image indices:",rnd_inds)

x_grid=[train_ds[i][0] for i in rnd_inds]
y_grid=[train_ds[i][1] for i in rnd_inds]

x_grid=utils.make_grid(x_grid, nrow=4, padding=2)
print(x_grid.shape)

## call helper function
plt.figure(figsize=(10,10))
show(x_grid,y_grid)

```

```python id="tUh4D1wYStUj"
from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)  
```

```python id="2Kj-bMJ4StUn" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533058393, "user_tz": -330, "elapsed": 19287, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="401a93d5-e3a4-46bf-b130-869817afa8cc"
## extract a batch from training data
for x, y in train_dl:
    print(x.shape)
    print(y.shape)
    break
```

```python id="LYyLmARFStUt" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533058395, "user_tz": -330, "elapsed": 19128, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ad79081a-eccf-4ac6-95e9-5ad9c109fcb2"
## extract a batch from validation data
for x, y in val_dl:
    print(x.shape)
    print(y.shape)
    break
```

<!-- #region id="1DGvPi--StU6" -->
## Building Model
<!-- #endregion -->

```python id="lCtl-NxOStU-"
from torchvision import models
import torch

## load model with random weights
model_resnet18 = models.resnet18(pretrained=False)
```

```python id="pWhDhQ1CStVG" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533110586, "user_tz": -330, "elapsed": 1115, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="11e2a629-6d51-4109-ad57-e203868fdcf7"
print(model_resnet18)
```

```python id="NzsFqDquStVL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533129034, "user_tz": -330, "elapsed": 11719, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6a289054-ae98-4e9f-ec62-d33dc2581539"
from torch import nn
## change the output layer
num_classes=10
num_ftrs = model_resnet18.fc.in_features 
model_resnet18.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda:0")
model_resnet18.to(device)
```

```python id="BW3O-T2gStVT"
from torchsummary import summary
summary(model_resnet18, input_size=(3, 224, 224))
```

```python id="_t3tEXngStVZ" colab={"base_uri": "https://localhost:8080/", "height": 390} executionInfo={"status": "ok", "timestamp": 1608533137076, "user_tz": -330, "elapsed": 1415, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f5186577-630f-4b7a-a1a3-e508035038c9"
## get Conv1 weights
for w in model_resnet18.parameters():
    w=w.data.cpu()
    print(w.shape)
    break

## normalize to [0,1]
min_w=torch.min(w)
w1 = (-1/(2*min_w))*w + 0.5 
print(torch.min(w1).item(),torch.max(w1).item())

## make a grid
grid_size=len(w1)
x_grid=[w1[i] for i in range(grid_size)]
x_grid=utils.make_grid(x_grid, nrow=8, padding=1)
print(x_grid.shape)

## call helper function
plt.figure(figsize=(5,5))
show(x_grid)
```

```python id="v4djNI6sStVi"
from torchvision import models
import torch

## load model with pretrained weights
resnet18_pretrained = models.resnet18(pretrained=True)

## change the output layer
num_classes=10
num_ftrs = resnet18_pretrained.fc.in_features
resnet18_pretrained.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda:0")
resnet18_pretrained.to(device) 
```

```python id="gkNZlQpiStVm" colab={"base_uri": "https://localhost:8080/", "height": 390} executionInfo={"status": "ok", "timestamp": 1608533145745, "user_tz": -330, "elapsed": 1279, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f2f097a0-cb5f-41ec-847c-a60fe1ce7422"
## get Conv1 weights
for w in resnet18_pretrained.parameters():
    w=w.data.cpu()
    print(w.shape)
    break

## normalize to [0,1]
min_w=torch.min(w)
w1 = (-1/(2*min_w))*w + 0.5 
print(torch.min(w1).item(),torch.max(w1).item())

## make a grid
grid_size=len(w1)
x_grid=[w1[i] for i in range(grid_size)]
x_grid=utils.make_grid(x_grid, nrow=8, padding=1)
print(x_grid.shape)

## call helper function
plt.figure(figsize=(5,5))
show(x_grid)
```

```python id="hrQBbeWPStVr" colab={"base_uri": "https://localhost:8080/", "height": 970, "referenced_widgets": ["4f0f8341e0e44ba1ad930f356f28958b", "8fc2b85f142c42939577a0a22f6d09c4", "8b769fb6af064814b63595046f0d1437", "876ba2a5c51d4dd29ff9bf1b9b2f1052", "ba1f8d42baf94012b6e9a70234f3314d", "e10bc77bb67b49ab8113128ca0cdc061", "4f3c030e4aa1489182e752682d6c3172", "c669b4bec4ac44f787acf4e3f4cb9697"]} executionInfo={"status": "ok", "timestamp": 1608533159550, "user_tz": -330, "elapsed": 9155, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dd2b62b4-dcb7-44bc-ef25-6396a2967e36"
num_classes=10
vgg19 = models.vgg19(pretrained=True)
## change the last layer
vgg19.classifier[6] = nn.Linear(4096,num_classes)
print(vgg19)
```

<!-- #region id="MU3qFsybStVw" -->
## Define Loss Function
<!-- #endregion -->

```python id="gQiKjdx9StVx"
loss_func = nn.CrossEntropyLoss(reduction="sum")
```

```python id="2f-4QZSLStV2" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533159557, "user_tz": -330, "elapsed": 6437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="00606404-e23f-4371-f161-64708705fc71"
## fix random seed
torch.manual_seed(0)

n,c=4,5
y = torch.randn(n, c, requires_grad=True)
print(y.shape)

loss_func = nn.CrossEntropyLoss(reduction="sum")
target = torch.randint(c,size=(n,))
print(target.shape)

loss = loss_func(y, target)
print(loss.item())
```

```python id="HqTt-J3fStV7" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533160268, "user_tz": -330, "elapsed": 6990, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1e9d39f6-0c1d-4d93-a887-a9eb1f727def"
loss.backward()
print (y.data)
```

<!-- #region id="bIz_gm_pStWE" -->
## Defining Optimizer
<!-- #endregion -->

```python id="YyV0_CqSStWF"
from torch import optim
opt = optim.Adam(model_resnet18.parameters(), lr=1e-4)
```

```python id="TNhJ8SiOStWJ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533160271, "user_tz": -330, "elapsed": 5360, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="91e02227-8a48-4465-9dc5-2743caf4fb10"
## get learning rate 
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

current_lr=get_lr(opt)
print('current lr={}'.format(current_lr))
```

```python id="O1FaQc2yStWQ"
from torch.optim.lr_scheduler import CosineAnnealingLR

## define learning rate scheduler
lr_scheduler = CosineAnnealingLR(opt,T_max=2,eta_min=1e-5)
```

```python id="ucx6_9F-StWX" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608533160272, "user_tz": -330, "elapsed": 5011, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c03bb3a0-e4bb-4b19-8ae7-3f891813fd6e"
lrs=[]
for i in range(10):
    lr_scheduler.step()
    lr=get_lr(opt)
    print("epoch %s, lr: %.1e" %(i,lr))
    lrs.append(lr)
```

```python id="qsAdc2SnStWd" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1608533160274, "user_tz": -330, "elapsed": 4843, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e50463ef-5e11-44f0-bbc5-c504909b5b5f"
plt.plot(lrs)
```

<!-- #region id="sHbYIiRhStWm" -->
## Training and Transfer Learning
<!-- #endregion -->

```python id="beZTY7H5StWo"
def metrics_batch(output, target):
    ## get output class
    pred = output.argmax(dim=1, keepdim=True)
    
    ## compare output class with target class
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects
```

```python id="6q-jDEezStWu"
def loss_batch(loss_func, output, target, opt=None):
    
    ## get loss 
    loss = loss_func(output, target)
    
    ## get performance metric
    metric_b = metrics_batch(output,target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b
```

```python id="xK9KPRWtStW1"
## define device as a global variable
device = torch.device("cuda")

def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data=len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        ## move batch to device
        xb=xb.to(device)
        yb=yb.to(device)
        
        ## get model output
        output=model(xb)
        
        ## get loss per batch
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        
        ## update running loss
        running_loss+=loss_b
        
        ## update running metric
        if metric_b is not None:
            running_metric+=metric_b

        ## break the loop in case of sanity check
        if sanity_check is True:
            break
    
    ## average loss value
    loss=running_loss/float(len_data)
    
    ## average metric value
    metric=running_metric/float(len_data)
    
    return loss, metric
```

```python id="jc5_HFYIStW9"
def train_val(model, params):
    ## extract model parameters
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    
    ## history of loss values in each epoch
    loss_history={
        "train": [],
        "val": [],
    }
    
    ## histroy of metric values in each epoch
    metric_history={
        "train": [],
        "val": [],
    }
    
    ## a deep copy of weights for the best performing model
    best_model_wts = copy.deepcopy(model.state_dict())
    
    ## initialize best loss to a large value
    best_loss=float('inf')
    
    ## main loop
    for epoch in range(num_epochs):
        
        ## get current learning rate
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        
        ## train model on training dataset
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)

        ## collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        ## evaluate model on validation dataset    
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
        
       
        ## store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            ## store weights into a local file
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
        
        ## collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        ## learning rate schedule
        lr_scheduler.step()

        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" %(train_loss,val_loss,100*val_metric))
        print("-"*10) 

    ## load best model weights
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history
```

<!-- #region id="9wpHS9IKStXF" -->
### Train With Random-Init Weights
<!-- #endregion -->

```python id="36mvnXO4StXJ"
import copy

loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(model_resnet18.parameters(), lr=1e-4)
lr_scheduler = CosineAnnealingLR(opt,T_max=5,eta_min=1e-6)

params_train={
 "num_epochs": 3,
 "optimizer": opt,
 "loss_func": loss_func,
 "train_dl": train_dl,
 "val_dl": val_dl,
 "sanity_check": False,
 "lr_scheduler": lr_scheduler,
 "path2weights": "./models/resnet18.pt",
}

## train and validate the model
model_resnet18,loss_hist,metric_hist=train_val(model_resnet18,params_train)
```

```python id="f20b7IA4StXN"
## Train-Validation Progress
num_epochs=params_train["num_epochs"]

## plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

## plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()
```

<!-- #region id="gMEjy5SmStXV" -->
### Train With Pre-Trained Weights
<!-- #endregion -->

```python id="6M8VKNffStXW"
import copy

loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(resnet18_pretrained.parameters(), lr=1e-4)
lr_scheduler = CosineAnnealingLR(opt,T_max=5,eta_min=1e-6)

params_train={
 "num_epochs": 3,
 "optimizer": opt,
 "loss_func": loss_func,
 "train_dl": train_dl,
 "val_dl": val_dl,
 "sanity_check": False,
 "lr_scheduler": lr_scheduler,
 "path2weights": "./models/resnet18_pretrained.pt",
}

## train and validate the model
resnet18_pretrained,loss_hist,metric_hist=train_val(resnet18_pretrained,params_train)
```

```python id="aNW5DTmQStXg"
## Train-Validation Progress
num_epochs=params_train["num_epochs"]

## plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

## plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()
```

<!-- #region id="t9HMH95GX4Rj" -->
## Deployment
<!-- #endregion -->

```python id="0_L8YMTNXs-7"
from torch import nn
from torchvision import models

## load model
model_resnet18 = models.resnet18(pretrained=False)
num_ftrs = model_resnet18.fc.in_features
## change last layer
num_classes=10
model_resnet18.fc = nn.Linear(num_ftrs, num_classes)
```

```python id="X1nKNRLoXs_H"
import torch 

## load state_dict into model
path2weights="./models/resnet18_pretrained.pt"
model_resnet18.load_state_dict(torch.load(path2weights))
```

```python id="x1S0tRTKXs_O"
## set model in evaluation mode
model_resnet18.eval();
```

```python id="UCk6DV9wXs_V"
## move model to cuda/gpu device
if torch.cuda.is_available():
    device = torch.device("cuda")
    model_resnet18=model_resnet18.to(device)
```

```python id="glmGeCAAXs_d"
def deploy_model(model,dataset,device, num_classes=10,sanity_check=False):

    len_data=len(dataset)
    
    ## initialize output tensor on CPU: due to GPU memory limits
    y_out=torch.zeros(len_data,num_classes)
    
    ## initialize ground truth on CPU: due to GPU memory limits
    y_gt=np.zeros((len_data),dtype="uint8")
    
    ## move model to device
    model=model.to(device)
    
    elapsed_times=[]
    with torch.no_grad():
        for i in range(len_data):
            x,y=dataset[i]
            y_gt[i]=y
            start=time.time()    
            yy=model(x.unsqueeze(0).to(device))
            y_out[i]=torch.softmax(yy,dim=1)
            elapsed=time.time()-start
            elapsed_times.append(elapsed)

            if sanity_check is True:
                break

    inference_time=np.mean(elapsed_times)*1000
    print("average inference time per image on %s: %.2f ms " %(device,inference_time))
    return y_out.numpy(),y_gt
```

<!-- #region id="YKKXZLRMXs_i" -->
### Loading Test Dataset
<!-- #endregion -->

```python id="PSuisl9-Xs_k" outputId="a988de23-8cae-46e4-8336-8f4559f4eaed"
from torchvision import datasets
import torchvision.transforms as transforms

## define transformation
data_transformer = transforms.Compose([transforms.ToTensor()])

path2data="./data"

## loading data
test0_ds=datasets.STL10(path2data, split='test', download=True,transform=data_transformer)
print(test0_ds.data.shape)
```

```python id="gVEe7PBlXs_u" outputId="92d541e7-ec59-4ceb-d063-78bcfccc689d"
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

indices=list(range(len(test0_ds)))
y_test0=[y for _,y in test0_ds]
for test_index, val_index in sss.split(indices, y_test0):
    print("test:", test_index, "val:", val_index)
    print(len(val_index),len(test_index))
```

```python id="Jg6qhJ6CXs_0"
from torch.utils.data import Subset

val_ds=Subset(test0_ds,val_index)
test_ds=Subset(test0_ds,test_index)
```

```python id="6dSCxX45Xs_9"
mean=[0.4467106, 0.43980986, 0.40664646]
std=[0.22414584,0.22148906,0.22389975]
```

```python id="fHmhFnBDXtAD"
test0_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])   
```

```python id="TfKcsa5gXtAK"
test0_ds.transform=test0_transformer
```

```python id="X5eq9x_LXtAS" outputId="810606df-528c-44f0-eb5e-09a0577eab21"
import time
import numpy as np

## deploy model 
y_out,y_gt=deploy_model(model_resnet18,val_ds,device=device,sanity_check=False)
print(y_out.shape,y_gt.shape)

```

```python id="Y60cd0UMXtAZ" outputId="b5185e0b-efc8-4f7f-9ad8-c7ffd617ee6f"
from sklearn.metrics import accuracy_score

## get predictions
y_pred = np.argmax(y_out,axis=1)
print(y_pred.shape,y_gt.shape)

## compute accuracy 
acc=accuracy_score(y_pred,y_gt)
print("accuracy: %.2f" %acc)

```

```python id="F77EyS8mXtAi" outputId="cb93328a-21dc-46ea-fddb-ea46214f54d3"
y_out,y_gt=deploy_model(model_resnet18,test_ds,device=device)

y_pred = np.argmax(y_out,axis=1)
acc=accuracy_score(y_pred,y_gt)
print(acc)
```

```python id="zr3JSaKQXtAo" outputId="af0d3702-552a-47eb-c797-e98db97761fb"
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
np.random.seed(1)

def imshow(inp, title=None):
    mean=[0.4467106, 0.43980986, 0.40664646]
    std=[0.22414584,0.22148906,0.22389975]
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  ## pause a bit so that plots are updated  

grid_size=4
rnd_inds=np.random.randint(0,len(test_ds),grid_size)
print("image indices:",rnd_inds)

x_grid_test=[test_ds[i][0] for i in rnd_inds]
y_grid_test=[(y_pred[i],y_gt[i]) for i in rnd_inds]

x_grid_test=utils.make_grid(x_grid_test, nrow=4, padding=2)
print(x_grid_test.shape)

plt.rcParams['figure.figsize'] = (10, 5)
imshow(x_grid_test,y_grid_test)
```

```python id="0xNvrL_qXtAw" outputId="c09e8471-f4d5-4ad1-f6fa-fdd6b0eae22f"
device_cpu = torch.device("cpu")
y_out,y_gt=deploy_model(model_resnet18,val_ds,device=device_cpu,sanity_check=False)
print(y_out.shape,y_gt.shape)
```
