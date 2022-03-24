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

```python id="Eei0NuRj35QP" executionInfo={"status": "ok", "timestamp": 1628780386303, "user_tz": -330, "elapsed": 1314, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-sor"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="UM90qwuS4K-k" executionInfo={"status": "ok", "timestamp": 1628780391097, "user_tz": -330, "elapsed": 4032, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="06d49ff2-c666-4a12-90ad-58a39d128fe6"
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

```python colab={"base_uri": "https://localhost:8080/"} id="TE_AZh_X4K-p" executionInfo={"status": "ok", "timestamp": 1628780626963, "user_tz": -330, "elapsed": 399, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1a93d0ce-62ae-4d7a-c832-b8fa82c64a25"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="sd_n1hSi4K-q" executionInfo={"status": "ok", "timestamp": 1628780629867, "user_tz": -330, "elapsed": 2212, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="75422aa2-85d1-44ad-d3dd-cf1a23c6b52f"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="p-GuT3uF4O9n" -->
---
<!-- #endregion -->

```python id="KeB5VRUd4TMn" executionInfo={"status": "ok", "timestamp": 1628780396331, "user_tz": -330, "elapsed": 4552, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix,  accuracy_score
import itertools
```

```python id="IGF0JS2V40y0" colab={"base_uri": "https://localhost:8080/", "height": 224} executionInfo={"status": "ok", "timestamp": 1628780396335, "user_tz": -330, "elapsed": 56, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="db8872d6-fff7-4745-acbb-447ba1cab4f5"
df = pd.read_csv('./data/silver/userdata.csv')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="pMmBLRi2zuwB" executionInfo={"status": "ok", "timestamp": 1628780396338, "user_tz": -330, "elapsed": 54, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d24673aa-564d-4958-8b57-da094a64811c"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 317} id="D6ztmfY0z4Cs" executionInfo={"status": "ok", "timestamp": 1628780396341, "user_tz": -330, "elapsed": 47, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7da04099-0458-430c-fda1-e45ced5f4d5a"
df.describe().round(1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 695} id="n00SxPNfzxSF" executionInfo={"status": "ok", "timestamp": 1628780399964, "user_tz": -330, "elapsed": 3667, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64890463-f07b-46a8-fcbf-762409ba0a12"
fig, ax = plt.subplots(figsize=(18,12))
df.hist(ax=ax)
plt.show()
```

<!-- #region id="N_n74jQ2z8m0" -->
## User Snapshot
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 193} id="J4TNwCwR0SzK" executionInfo={"status": "ok", "timestamp": 1628780399970, "user_tz": -330, "elapsed": 40, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9ae2a107-4638-43b6-b693-8f160ccbd573"
df[df.id == 'e12aeaf2d47d42479ea1c4ac3d8286c6']
```

<!-- #region id="ATh2mJed0Tvr" -->
user completed an offer 0b1e... and viewed ae26.... Offer 2906.. had been ignored twice.
<!-- #endregion -->

<!-- #region id="x2n08lpk0uXr" -->
## Train Embeddings
<!-- #endregion -->

```python id="qxIgwSbO0lcD" executionInfo={"status": "ok", "timestamp": 1628780399973, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def to_categorical(df, columns):
    for col in columns:
        df[col] = pd.Categorical(df[col])
        df[col] = df[col].cat.codes
    return df
```

```python id="VnuKTtyW0qPZ" executionInfo={"status": "ok", "timestamp": 1628780399975, "user_tz": -330, "elapsed": 38, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Set embedding sizes
N = len(df['id'].unique())
M = len(df['offer_id'].unique())

# Set embedding dimension
D = 100
```

```python id="vOEAz79Y0wL1" executionInfo={"status": "ok", "timestamp": 1628780399977, "user_tz": -330, "elapsed": 38, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Create a neural network
class Model(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, output_dim, layers=[1024], p=0.4):
        super(Model, self).__init__()
        self.N = n_users
        self.M = n_items
        self.D = embed_dim

        self.u_emb = nn.Embedding(self.N, self.D)
        self.m_emb = nn.Embedding(self.M, self.D)
        
        layerlist = []
        n_in = 2 * self.D
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU())
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],output_dim))
        self.layers = nn.Sequential(*layerlist)
        
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, u, m):
        u = self.u_emb(u) # output is (num_samples, D)
        m = self.m_emb(m) # output is (num_samples, D)

        # merge
        out = torch.cat((u, m), 1) # output is (num_samples, 2D)

        x = self.layers(out)
        return x
```

```python colab={"base_uri": "https://localhost:8080/"} id="idiRb2_U0yLP" executionInfo={"status": "ok", "timestamp": 1628780412357, "user_tz": -330, "elapsed": 12416, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9cfb08be-0244-4866-dd63-06ac8d56804d"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Model(N, M, D, output_dim=df['event'].nunique(), layers=[512, 256])
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)
```

```python id="8a-7ysuD16EU" executionInfo={"status": "ok", "timestamp": 1628780412359, "user_tz": -330, "elapsed": 75, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df = to_categorical(df, ['id','offer_id'])
```

```python id="DJtFuLZEzMRl" executionInfo={"status": "ok", "timestamp": 1628780412360, "user_tz": -330, "elapsed": 70, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Convert to tensors
user_ids_t = torch.from_numpy(df['id'].values).long()
offer_ids_t = torch.from_numpy(df['offer_id'].values).long()
ratings_t = torch.from_numpy(df['event'].values).long()
```

```python id="P2f5p0OezMRm" executionInfo={"status": "ok", "timestamp": 1628780412362, "user_tz": -330, "elapsed": 69, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Make datasets
N_train = int(0.8 * len(df['event'].values))
N_test = 1000
train_dataset = torch.utils.data.TensorDataset(
    user_ids_t[:N_train],
    offer_ids_t[:N_train],
    ratings_t[:N_train],
)

val_dataset = torch.utils.data.TensorDataset(
    user_ids_t[N_train:-N_test],
    offer_ids_t[N_train:-N_test],
    ratings_t[N_train:-N_test],
)
test_df = df[-N_test:]
```

```python id="eBbbAHTmzMRo" executionInfo={"status": "ok", "timestamp": 1628780412364, "user_tz": -330, "elapsed": 70, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
```

```python id="6loeuzQezMRp" executionInfo={"status": "ok", "timestamp": 1628780412366, "user_tz": -330, "elapsed": 69, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# A function to encapsulate the training loop
def batch_gd(model, criterion, optimizer, train_iter, test_iter, epochs):
    
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    acc_list = []
    
    for it in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for users, offer, targets in train_loader:
            

            # move data to GPU
            users, offer, targets = users.to(device), offer.to(device), targets.to(device)
            #targets = targets.view(-1, 1).long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(users, offer)
            
            loss = criterion(outputs, targets.squeeze())

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            
            # Track the accuracy
            total = targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            acc = correct / total
            acc_list.append(acc)

        # Get train loss and test loss
        train_loss = np.mean(train_loss) # a little misleading
        
        val_loss = []
        
        # validation
        with torch.no_grad():
            model.eval()
            
            for users, offer, targets in validation_loader:
                users, offer, targets = users.to(device), offer.to(device), targets.to(device)
                #targets = targets.view(-1, 1).long()
                outputs = model(users, offer)
                loss = criterion(outputs, targets.squeeze())
                val_loss.append(loss.item())
        
        val_loss = np.mean(val_loss)
        # Save losses
        train_losses[it] = train_loss
        val_losses[it] = val_loss

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Validation Loss: {train_loss:.4f}, '
              f'Test Loss: {val_loss:.4f}, Accuracy: {acc}, Duration: {dt}')

    return train_losses, val_losses
```

```python id="bD29bP1jzMRr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628780559011, "user_tz": -330, "elapsed": 146707, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="62b87be1-ca3f-4a71-f591-8d2a45d1c0d1"
train_losses, val_losses = batch_gd(model, criterion, optimizer, train_loader, validation_loader, 25)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="jbPcO6qs2q5L" executionInfo={"status": "ok", "timestamp": 1628780559013, "user_tz": -330, "elapsed": 35, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cf3aadc4-c6df-4778-a667-55905c847ecc"
# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='test loss')
plt.legend()
plt.show()
```

```python id="StByZmyTzMRs" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628780559872, "user_tz": -330, "elapsed": 888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="40a86463-3c5e-43d6-a07e-cdd5c37f0ab3"
start_ix = 10
end_ix = 20
test_X =  torch.from_numpy(test_df.iloc[start_ix:end_ix]['id'].values).long()
test_y = torch.from_numpy(test_df.iloc[start_ix:end_ix]['event'].values).long()

with torch.no_grad():
    model.to('cpu')
    model.eval()
    pred = model(test_X, test_y)
    print(pred)

_, predicted = torch.max(pred.data, 1)
print(predicted)
```

```python id="PoDbqNp4zMRv" executionInfo={"status": "ok", "timestamp": 1628780559875, "user_tz": -330, "elapsed": 59, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
```

<!-- #region id="Oze7QdiAzMRx" -->
#### Plot confusion matrix and baseline accuracy
<!-- #endregion -->

```python id="Nqdi8VHbzMRx" colab={"base_uri": "https://localhost:8080/", "height": 362} executionInfo={"status": "ok", "timestamp": 1628780559877, "user_tz": -330, "elapsed": 59, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2bb7c7e7-85cf-4e24-f3ab-9f2b50b62ff6"
cm = confusion_matrix(test_y, predicted)
classes = [0,1,2]
plot_confusion_matrix(cm, classes)
```

```python id="_GJ9RHUzzMRy" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628780559879, "user_tz": -330, "elapsed": 53, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="22a87eae-fe76-4773-83bc-d5c3a57e759c"
print("Accuracy so far: " + str(100*accuracy_score(test_y, predicted))+ "%" )
```

<!-- #region id="PeSiLjeMzMRz" -->
Results are decent so far and almost twice better than random quessing.

#### Show some misclassified examples
<!-- #endregion -->

```python id="Wroj_OulzMR0" colab={"base_uri": "https://localhost:8080/", "height": 359} executionInfo={"status": "ok", "timestamp": 1628780559887, "user_tz": -330, "elapsed": 53, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1b9008b3-2b8e-4874-cdaf-111c7164a14a"
data = test_df.iloc[start_ix:end_ix][['age', 'became_member_on', 'gender', 'id', 'income', 'memberdays', 'event']]#['offer_id'].values
pred_values = pd.DataFrame(predicted, columns=['predicted'], index=data.index)
pd.concat([data, pred_values], axis=1)
```

<!-- #region id="4dhRmFWazMR2" -->
Now let's save the model for future reference
<!-- #endregion -->

```python id="YvugotPuzMR2" executionInfo={"status": "ok", "timestamp": 1628780618014, "user_tz": -330, "elapsed": 616, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def save_model(model, model_name, model_info):
    # Save the parameters used to construct the model
    with open(model_name, 'wb') as f:
        torch.save(model_info, f)

    # Save the model parameters
    
    with open(model_name, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

model_info = {
         'n_users': M, 
          'n_items': N, 
          'embed_dim': D, 
          'output_dim': df['event'].nunique(), 
          'layers': [512, 256], 
          'p': 0.4
    }
save_model(model, './artifacts/models/BaselineModel.pth', model_info)
```

<!-- #region id="Fa_Zwy5dzMR4" -->
During the next step we improve the model to take additional paramenters that describe each user and each offer, which should hopefully, give the model insigths on why a particular customer may like or not like given offer.
<!-- #endregion -->
