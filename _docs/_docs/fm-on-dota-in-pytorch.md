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

<!-- #region id="OM759O-dhfqc" -->
# Training Factorization Machine (FM) RecSys Model on Dota dataset in PyTorch
<!-- #endregion -->

```python id="v-xvDcvNgODS" executionInfo={"status": "ok", "timestamp": 1638102290009, "user_tz": -330, "elapsed": 6873, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

import warnings, random
warnings.filterwarnings('ignore')
```

```python colab={"base_uri": "https://localhost:8080/"} id="0bnnoD8-gSE5" executionInfo={"status": "ok", "timestamp": 1638102290017, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="262f543a-9dea-456c-a2cf-865e584c6ed8"
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
```

```python id="L1zo63KAgUDp" executionInfo={"status": "ok", "timestamp": 1638102294721, "user_tz": -330, "elapsed": 467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

```python colab={"base_uri": "https://localhost:8080/"} id="4s2kvqTBgW2U" executionInfo={"status": "ok", "timestamp": 1638102322897, "user_tz": -330, "elapsed": 1220, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2c246804-099e-4e00-a9d7-37be696dc0d4"
!wget -q --show-progress https://github.com/RecoHut-Datasets/dota/raw/v1/dota-heroes-binary.zip
!unzip dota-heroes-binary.zip
```

```python id="JT7XjsKBgVXC" executionInfo={"status": "ok", "timestamp": 1638102392759, "user_tz": -330, "elapsed": 1398, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
train_df = pd.read_csv('/content/dota_train_binary_heroes.csv')
test_df = pd.read_csv('/content/dota_test_binary_heroes.csv')
target = pd.read_csv('/content/train_targets.csv')

train_df.set_index('match_id_hash', inplace=True)
test_df.set_index('match_id_hash', inplace=True)
target.set_index('match_id_hash', inplace=True)
```

```python id="kiX8c2_ugzh4" executionInfo={"status": "ok", "timestamp": 1638102424360, "user_tz": -330, "elapsed": 489, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
x_train, x_test, y_train, y_test = train_test_split(train_df.values, target['radiant_win'].values, test_size=0.2, random_state=1995)
```

```python id="vc_uVxmwg07x" executionInfo={"status": "ok", "timestamp": 1638102429778, "user_tz": -330, "elapsed": 613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Config:
    learning_rate = 0.01/2
    weight_decay = 0.1/2
    early_stopping_round = 0
    epochs = 30
    seed = 1995
    dim_f = 20
    alpha = 100
    batch_size = 64
    
config = Config()
```

```python id="d34tSZl9g2Py" executionInfo={"status": "ok", "timestamp": 1638102447095, "user_tz": -330, "elapsed": 520, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Dataset(Dataset):
    def __init__(self, data, target, train):
        self.data = data
        self.train = train
        
        if train:
            self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.train:
            return {
                'x' : torch.tensor(self.data[idx, :], dtype=torch.float).to(device),
                'y' : torch.tensor(self.target[idx], dtype=torch.float).to(device)
            }
        else:
            return {
                'x' : torch.tensor(self.data[idx, :], dtype=torch.float).to(device),
            }
```

```python id="5KKUdKCIg3y9" executionInfo={"status": "ok", "timestamp": 1638102456817, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class FM(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(FM, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, dtype=torch.float))
        self.V = nn.Parameter(torch.randn((input_dim, embed_dim), dtype=torch.float))
        self.bias = nn.Parameter(torch.tensor([1], dtype=torch.float))

    def forward(self, batch_data):
        feature_effect = torch.matmul(batch_data, self.W)
        temp = torch.matmul(batch_data, self.V)**2 - torch.matmul(batch_data**2, self.V**2)
        interaction_effect = torch.sum(temp, axis=1) / 2     

        return self.bias + feature_effect + interaction_effect
```

```python colab={"base_uri": "https://localhost:8080/"} id="ksMV0kaEg8iM" executionInfo={"status": "ok", "timestamp": 1638102525034, "user_tz": -330, "elapsed": 30981, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="104b3f24-06a2-4f5b-b1c8-41708f8719bb"
seed_everything(config.seed)

train_dataset = Dataset(x_train, y_train, train=True)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

test_dataset = Dataset(x_test, y_test, train=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size*100, shuffle=False, drop_last=False)

model = FM(input_dim=train_df.shape[1],
           embed_dim=config.dim_f)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
loss_fn = nn.BCEWithLogitsLoss()

start = datetime.now()
history = defaultdict(list)
history['best_loss'] = np.inf

for epoch in range(config.epochs):
    
    model.train()
    losses = 0

    for batch_data in train_loader:
        x = batch_data['x']
        y = batch_data['y']

        optimizer.zero_grad()
        
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        losses += loss.item()
    losses /= len(train_loader) 
    history['train_losses'].append(losses)
    
    losses_val = 0

    for bacth_data in test_loader:
        x = batch_data['x']
        y = batch_data['y']

        with torch.no_grad():

            pred = model(x)
            loss = loss_fn(pred, y)
            losses_val += loss.item()
    
    losses_val /= len(test_loader)
    history['val_losses'].append(losses_val)
    print(f'EPOCH {epoch+1} TRAIN LogLoss : {losses:.6f}, TEST LogLoss : {losses_val:.6f}')
    
    if history['best_loss'] > losses_val:
        history['best_loss'] = losses_val
        # torch.save(model.state_dict(), f'../paper_review/4. Deep learning based/FM/FM.pth')
        print('The Model Saving...')
    # if epoch==0 or (epoch + 1) % 10 == 0 or epoch == config.epochs:
    
end = datetime.now()
print(f'Training takes time {end-start}')
```

<!-- #region id="dn62FloshW0i" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FO0hoDUAhW0n" executionInfo={"status": "ok", "timestamp": 1638102580318, "user_tz": -330, "elapsed": 3364, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="32dcc1d8-3958-41e1-f8e9-b6b123ccbd52"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="WJwQdCG_hW0o" -->
---
<!-- #endregion -->

<!-- #region id="6ZpHElishW0p" -->
**END**
<!-- #endregion -->
