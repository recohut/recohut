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

```python
import os
project_name = "reco-tut-sor"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python
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

```python
!git status
```

```python
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

---

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix,  accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
import itertools
```

```python
df = pd.read_csv('./data/silver/userdata.csv')
df.head()
```

```python
def to_categorical(df, columns):
    for col in columns:
        df[col] = pd.Categorical(df[col])
        df[col] = df[col].cat.codes
    return df
```

```python
df = to_categorical(df, ['id', 'offer_id', 'event'])
df.dtypes
```

Recommendation matrix is very similar to embeddings. So we will leverage this and will train embedding along the model.

```python
# Set embedding sizes
N = len(df['id'].unique())
M = len(df['offer_id'].unique())

# Set embedding dimension
D = 20
```

```python
offer_specs = ['difficulty', 'duration', 'reward', 'web',
       'email', 'mobile', 'social', 'bogo', 'discount', 'informational']
user_specs = ['age', 'became_member_on', 'gender', 'income', 'memberdays']
```

```python
# Create a neural network that takes additional continuous paramets
class Model(nn.Module):
    def __init__(self, n_users, n_items, n_cont_user, n_cont_offer, embed_dim, output_dim, layers=[1024], p=0.4):
        super(Model, self).__init__()
        self.N = n_users
        self.M = n_items
        self.D = embed_dim
        self.bn_cont_u = nn.BatchNorm1d(n_cont_user)
        self.bn_cont_o = nn.BatchNorm1d(n_cont_offer)
        self.u_emb = nn.Embedding(self.N, self.D)
        self.m_emb = nn.Embedding(self.M, self.D)
        
        layerlist = []
        n_in = 2 * self.D + n_cont_user + n_cont_offer
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            layerlist.append(nn.ReLU())
            n_in = i
        layerlist.append(nn.Linear(layers[-1],output_dim))
        self.layers = nn.Sequential(*layerlist)
        
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, user, offer, user_details, offer_details):
        u = self.u_emb(user) # output is (num_samples, D)
        m = self.m_emb(offer) # output is (num_samples, D)

        # merge
        out = torch.cat((u, m), 1) # output is (num_samples, 2D)
        u_cont = self.bn_cont_u(user_details)
        o_cont = self.bn_cont_o(offer_details)
        out = torch.cat([out, u_cont, o_cont], 1)
        x = self.layers(out)
        return x
```

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

layers = [1024, 1024, 512, 256, 128]
model = Model(N, M, 
              n_cont_user=df[user_specs].shape[1], 
              n_cont_offer=df[offer_specs].shape[1],
              embed_dim=D, 
              output_dim=df['event'].nunique(), 
              layers=layers)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9)
```

```python
model
```

### Create additional user and offer details tensors

```python
# Convert to tensors
# Make datasets
# We name events as rating for give better insight on the target value
# and ease of comparison to other similar models

user_ids_t = torch.from_numpy(df['id'].values).long()
offer_ids_t = torch.from_numpy(df['offer_id'].values).long()
ratings_t = torch.from_numpy(df['event'].values).long()

user_params_t = torch.from_numpy(df[user_specs].values).float()
offer_params_t = torch.from_numpy(df[offer_specs].values).float()
```

```python
# Make datasets
N_train = int(0.8 * len(df['event'].values))
N_test = 1000
train_dataset = torch.utils.data.TensorDataset(
    user_ids_t[:N_train],
    offer_ids_t[:N_train],
    user_params_t[:N_train],
    offer_params_t[:N_train],
    ratings_t[:N_train],
)

val_dataset = torch.utils.data.TensorDataset(
    user_ids_t[N_train:-N_test],
    offer_ids_t[N_train:-N_test],
    user_params_t[N_train:-N_test],
    offer_params_t[N_train:-N_test],
    ratings_t[N_train:-N_test],
    
)
test_df = df[-N_test:]
```

```python
# Data loaders
batch_size = 8
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
```

```python
# A function to encapsulate the training loop
def batch_gd(model, criterion, optimizer, train_iter, test_iter, epochs):
    
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    acc_list = []
    
    for it in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for users, offers, u_params, o_params, targets in train_loader:
            

            # move data to GPU
            users, offers, u_params, o_params, targets = users.to(device), offers.to(device),  u_params.to(device), o_params.to(device), targets.to(device)
            #targets = targets.view(-1, 1).long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(users, offers, u_params, o_params)
            
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
        train_loss = np.mean(train_loss)
        
        val_loss = []
        
        # validation
        with torch.no_grad():
            model.eval()
            
            for users, offers, u_params, o_params, targets in validation_loader:
                users, offers, u_params, o_params, targets = users.to(device), offers.to(device),  u_params.to(device), o_params.to(device), targets.to(device)
                #targets = targets.view(-1, 1).long()
                outputs = model(users, offers, u_params, o_params)
                loss = criterion(outputs, targets.squeeze())
                val_loss.append(loss.item())
        
        val_loss = np.mean(val_loss)
        # Save losses
        train_losses[it] = train_loss
        val_losses[it] = val_loss

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, '
              f'Validations Loss: {val_loss:.4f}, Accuracy: {acc:.4f}, Duration: {dt}')

    return train_losses, val_losses
```

```python
train_losses, val_losses = batch_gd(model, criterion, optimizer, train_loader, validation_loader, 35)
```

```python
# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='test loss')
plt.legend()
plt.show()
```

```python
start_ix = 0
end_ix = -1
test_X =  torch.from_numpy(test_df.iloc[start_ix:end_ix]['id'].values).long()
test_y = torch.from_numpy(test_df.iloc[start_ix:end_ix]['event'].values).long()
user_params_t = torch.from_numpy(test_df.iloc[start_ix:end_ix][user_specs].values).float()
offer_params_t = torch.from_numpy(test_df.iloc[start_ix:end_ix][offer_specs].values).float()

with torch.no_grad():
    model.eval()
    pred = model(test_X.to(device), test_y.to(device), user_params_t.to(device), offer_params_t.to(device))
    #print(pred)
    
_, predicted = torch.max(pred.data, 1)
print(predicted)
```

```python
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save=False,
                          figname='cm.png'):
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
    
    fig = plt.figure()
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
    if save:
        plt.savefig(figname, dpi=fig.dpi)
    plt.show()
```

#### Plot confusion matrix and baseline accuracy

```python
# Tensors should be copied back to cpu using tensor.cpu()
test_y = test_y.cpu()
predicted = predicted.cpu()
cm = confusion_matrix(test_y, predicted)
classes = [0,1,2]
plot_confusion_matrix(cm, classes, save=True, figname='./outputs/RecommendationEngine-cm.png')
```

```python
print("Accuracy so far: " + str(100*accuracy_score(test_y, predicted))+ "%" )
```

Results are decent so far and almost twice better than random quessing.

#### Show some predicted examples

```python
start_ix = 10
end_ix = 30
data = test_df.iloc[start_ix:end_ix][['age', 'became_member_on', 'gender', 'id', 'income', 'memberdays', 'event']]#['offer_id'].values
pred_values = pd.DataFrame(predicted[start_ix:end_ix], columns=['predicted'], index=data.index)
results = pd.concat([data, pred_values], axis=1)
results
```

From the output we see that for randomly selected 10 users model was inaccurate twice, but was able to predict user positive actions on offer.

Mislassifications:

```python
results[results.event != results.predicted]
```

Now let's save the model for future reference

```python
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
          'n_cont_user': df[user_specs].shape[1], 
          'n_cont_offer': df[offer_specs].shape[1],
          'embed_dim': D, 
          'output_dim': df['event'].nunique(), 
          'layers': layers, 
          'p': 0.4
    }
save_model(model, './artifacts/models/RecommendationModel2.pth', model_info)
```

Let's additionally check another metrics for the model

```python
layers = [1024, 1024, 512, 256, 128]
model2 = Model(N, M, 
              n_cont_user=df[user_specs].shape[1], 
              n_cont_offer=df[offer_specs].shape[1],
              embed_dim=D, 
              output_dim=df['event'].nunique(), 
              layers=layers)

model2.load_state_dict(torch.load('./artifacts/models/RecommendationModel2.pth'));
model2.to(device).eval()
```

```python
start_ix = 0
end_ix = -1
test_X =  torch.from_numpy(test_df.iloc[start_ix:end_ix]['id'].values).long()
test_y = torch.from_numpy(test_df.iloc[start_ix:end_ix]['event'].values).long()
user_params_t = torch.from_numpy(test_df.iloc[start_ix:end_ix][user_specs].values).float()
offer_params_t = torch.from_numpy(test_df.iloc[start_ix:end_ix][offer_specs].values).float()

with torch.no_grad():
    model2.eval()
    pred = model2(test_X.to(device), test_y.to(device), user_params_t.to(device), offer_params_t.to(device))
    #print(pred)
_, predicted = torch.max(pred.data, 1)
#print(predicted)
```

```python
# Tensors should be copied back to cpu using tensor.cpu()
test_y = test_y.cpu()
predicted = predicted.cpu()
cm = confusion_matrix(test_y, predicted)
classes = [0,1,2]
plot_confusion_matrix(cm, classes)
```

```python
print("Accuracy so far: " + str(100*accuracy_score(test_y, predicted))+ "%" )
```

```python
print("F1 score for the model: " + str(f1_score(test_y, predicted, average='weighted')))
print("Recall score for the model: " + str(recall_score(test_y, predicted, average='weighted')))
print("Precision score for the model: " + str(precision_score(test_y, predicted, average='weighted')))
```
