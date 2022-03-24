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

<!-- #region id="gh25Ewd3kust" -->
## Setup

Download raw data, import pytorch and other python libraries.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="joAvJDL8hIie" executionInfo={"status": "ok", "timestamp": 1637146618868, "user_tz": -330, "elapsed": 8500, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9423712b-58be-4eb9-ec86-bb78fd7761fd"
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/train.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/item_info.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track1_testset.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track2_testset.parquet.snappy
```

```python id="gjC-ZhlLhNsw"
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
```

<!-- #region id="X_p5glspkwBe" -->
## Model

Create an Pytorch-based MLP model, run the model on small synthetic data for sanity check.
<!-- #endregion -->

```python id="my92_80ph0oh"
class VanillaBaseModel(nn.Module):
    def __init__(self, 
                 num_items,
                 dim_item_emb=64,
                 dim_item_discrete_feature_emb=16,
                 dim_user_discrete_feature_emb=16,
                ):
        super().__init__()
        self.NUM_ITEM_DISCRETE_FEATURE = 3 + 1 # item_vec3 + location1
        self.NUM_ITEM_CONT_FEATURE = 2 + 1 # item_vec2 + price1
        self.NUM_USER_DISCRETE_FEATURE = 10
        self.dim_item_emb = dim_item_emb

        self.item_emb = nn.Embedding(num_items + 1, dim_item_emb) # num_items + 1
        
        # item discrete feature
        self.item_discrete_feature_emb_list = nn.ModuleList()
        num_unique_value_list = [4, 10, 2, 3]
        for i in range(self.NUM_ITEM_DISCRETE_FEATURE):
            num_unique_value = num_unique_value_list[i]
            self.item_discrete_feature_emb_list.append(
                nn.Embedding(num_unique_value, dim_item_discrete_feature_emb)
            )
        
        # user discrete feature
        self.user_discrete_feature_emb_list = nn.ModuleList()
        num_unique_value_list = [3, 1430, 20, 10, 198, 52, 3, 13, 2, 2347]
        for i in range(self.NUM_USER_DISCRETE_FEATURE):
            num_unique_value = num_unique_value_list[i]
            self.user_discrete_feature_emb_list.append(
                nn.Embedding(num_unique_value, dim_user_discrete_feature_emb)
            )

        # backbone
        self.backbone = nn.Sequential(
            nn.Linear(dim_item_emb + # user_click_history
                      self.NUM_ITEM_DISCRETE_FEATURE * dim_item_discrete_feature_emb + 
                      self.NUM_ITEM_CONT_FEATURE + 
                      self.NUM_USER_DISCRETE_FEATURE * dim_user_discrete_feature_emb +
                      dim_item_emb, 200), 
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.LayerNorm(200),
            nn.Linear(200, 80),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(80),
            nn.Linear(80, 1)
        )


    def forward(self,
                user_click_history,
                num_user_click_history,
                user_discrete_feature,
                item_id,
                item_discrete_feature,
                item_cont_feature
                ):
        """
        user_click_history: [N, 300]
        num_user_click_history: [N, 1]
        user_discrete_feature: [N, 10]
        item_id: [N, 1]
        item_discrete_feature: [N, 3 + 1], item_vec3 + location1
        item_cont_feature: [N, 2 + 1], item_vec2 + price1
        """

        batch_size = user_click_history.size()[0]
    
        # user click history emb
        tmp = self.item_emb(user_click_history) # [N, 300] -> [N, 300, dim_item_emb]
        user_click_history_emb = torch.zeros((batch_size, self.dim_item_emb))
        for i in range(batch_size):
            #print(num_user_click_history[i])
            aa = tmp[i, :num_user_click_history[i], :] # [N, D]
            #print(aa.shape)
            a = torch.mean(aa, dim=0) # [N, d] -> [1, d]
            #print(a.shape)
            #print(user_click_history_emb.shape)
            user_click_history_emb[i] = a

        ## User Profile Features
        # user discrete feature, 10 features
        tmp = []
        for i in range(self.NUM_USER_DISCRETE_FEATURE):
            tmp.append(
                self.user_discrete_feature_emb_list[i](user_discrete_feature[:, i]) # [N, dim_user_discrete_feature_emb]
            )
        user_discrete_feature_emb = torch.cat(tmp, dim=1)

        ## Item
        # item discrete feature, 3 features
        tmp = []
        for i in range(self.NUM_ITEM_DISCRETE_FEATURE):
            # print(i)
            # print(item_discrete_feature[:, i])
            tmp.append(
                self.item_discrete_feature_emb_list[i](item_discrete_feature[:, i]) # [N, dim_user_discrete_feature_emb]
            )
        item_discrete_feature_emb = torch.cat(tmp, dim=1)
        # item emb
        item_emb = self.item_emb(item_id)
        item_emb = torch.squeeze(item_emb)

        ## all emb
        #print(user_click_history_emb.size())
        #print(user_discrete_feature_emb.size())
        #print(item_discrete_feature_emb.size())
        #print(item_cont_feature.size())
        #print(item_emb.size())

        all_emb = torch.cat([user_click_history_emb, 
                             user_discrete_feature_emb,
                             item_discrete_feature_emb,
                             item_cont_feature,
                             item_emb,
                            ], dim=1) # [N, D]
        
        out = self.backbone(all_emb) # [N, 1]
        return out
```

```python id="ExD5fGsYh_4_"
m = VanillaBaseModel(
    num_items=381,
    dim_item_emb=64,
    dim_item_discrete_feature_emb=16,
    dim_user_discrete_feature_emb=16,
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="4LIu-4Xrh_2A" executionInfo={"status": "ok", "timestamp": 1637146465834, "user_tz": -330, "elapsed": 432, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8399ce52-49c1-4da5-b102-a32b7fdeb09c"
B = 3
m(
    user_click_history=torch.ones([B, 300], dtype=torch.int32),
    num_user_click_history=torch.ones([B, 1], dtype=torch.int32) * 10,
    user_discrete_feature=torch.ones([B, 10], dtype=torch.int32),
    item_id=torch.ones([B, 1], dtype=torch.int32),
    item_discrete_feature=torch.ones([B, 4], dtype=torch.int32),
    item_cont_feature=torch.randn([B, 3]),
)
```

<!-- #region id="SlrURkjKcnjU" -->
## Accumulated Indexing

Create a list of accumulated user portrait ids, and a list of portrait_id to id dictionary mapping.
<!-- #endregion -->

```python id="CQPffjz6iiQQ"
portraitidx_to_idx_dict_list = []
for i in range(10):
    portraitidx_to_idx_dict_list.append(dict())
acculumated_idx = [0] * 10
```

```python colab={"base_uri": "https://localhost:8080/"} id="JqapzW5NikBs" executionInfo={"status": "ok", "timestamp": 1637146688053, "user_tz": -330, "elapsed": 10848, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8e62ce4b-8261-43cd-c7e4-37819c593b47"
df_train = pd.read_parquet('train.parquet.snappy')
for i in tqdm(range(df_train.shape[0])):
    user_portrait = [int(s) for s in df_train.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
            acculumated_idx[idx] += 1

df_test1 = pd.read_parquet('track1_testset.parquet.snappy')
for i in tqdm(range(df_test1.shape[0])):
    user_portrait = [int(s) for s in df_test1.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
            acculumated_idx[idx] += 1

df_test2 = pd.read_parquet('track2_testset.parquet.snappy')
for i in tqdm(range(df_test2.shape[0])):
    user_portrait = [int(s) for s in df_test2.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
            acculumated_idx[idx] += 1

acculumated_idx
```

<!-- #region id="gpTXxl8skAde" -->
## Dataloader
<!-- #endregion -->

```python id="1tBra0avi68f"
def load_data():
    # item info
    df_item_info = pd.read_parquet('item_info.parquet.snappy')
    item_info_dict = {}
    for i in tqdm(range(df_item_info.shape[0])):
        item_id = df_item_info.at[i, 'item_id'] 

        item_discrete = df_item_info.at[i, 'item_vec'].split(',')[:3]
        item_cont = df_item_info.at[i, 'item_vec'].split(',')[-2:]
        price = df_item_info.at[i, 'price'] / 3000
        loc = df_item_info.at[i, 'location'] - 1 # 0~2

        item_cont.append(price) # 2 + 1
        item_discrete.append(loc) # 3 + 1

        item_cont = [float(it) for it in item_cont]
        item_discrete = [int(it) for it in item_discrete]
        item_discrete[0] = item_discrete[0] - 1 # 1~4 -> 0~3
        item_discrete[2] = item_discrete[2] - 1 # 1~2 -> 0~1

        item_info_dict[int(item_id)] = {
            'cont': np.array(item_cont, dtype=np.float64),
            'discrete': np.array(item_discrete, dtype=np.int64),
        }

    # trainset
    train_samples = []
    val_samples = []
    df_train = pd.read_parquet('train.parquet.snappy')

    # shuffle
    df_train = shuffle(df_train, random_state=2333).reset_index()
    total_num = int(df_train.shape[0])
    num_train = int(total_num * 0.95)
    num_val = total_num - num_train

    for i in tqdm(range(total_num)):
        if df_train.at[i, 'user_click_history'] == '0:0':
            user_click_list = [0]
        else:
            user_click_list = df_train.at[i, 'user_click_history'].split(',')
            user_click_list = [int(sample.split(':')[0]) for sample in user_click_list]
        num_user_click_history = len(user_click_list)
        tmp = np.zeros(400, dtype=np.int64)
        tmp[:len(user_click_list)] = user_click_list
        user_click_list = tmp
        
        exposed_items = [int(s) for s in df_train.at[i, 'exposed_items'].split(',')]
        labels = [int(s) for s in df_train.at[i, 'labels'].split(',')]

        user_portrait = [int(s) for s in df_train.at[i, 'user_protrait'].split(',')]
        # portraitidx_to_idx_dict_list: list of 10 dict, int:int
        for j in range(10):
            user_portrait[j] = portraitidx_to_idx_dict_list[j][user_portrait[j]]
        for k in range(9):
            one_sample = {
                'user_click_list': user_click_list,
                'num_user_click_history': num_user_click_history,
                'user_portrait': np.array(user_portrait, dtype=np.int64),
                'item_id': exposed_items[k],
                'label': labels[k]
            }
            if i < num_train:
                train_samples.append(one_sample)
            else:
                val_samples.append(one_sample)
    return item_info_dict, train_samples, val_samples
```

```python id="4tb4yDY4jaWW"
class BigDataCupDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 item_info_dict,
                 database
                ):
        super().__init__()
        self.item_info_dict = item_info_dict
        self.database = database

    def __len__(self, ):
        return len(self.database)

    def __getitem__(self, idx):
        one_sample = self.database[idx]
        user_click_history = one_sample['user_click_list']
        num_user_click_history = one_sample['num_user_click_history']
        user_discrete_feature = one_sample['user_portrait']
        item_id = one_sample['item_id']
        item_discrete_feature = self.item_info_dict[item_id]['discrete']
        item_cont_feature = self.item_info_dict[item_id]['cont']
        label = one_sample['label']

        # print(num_user_click_history)

        user_click_history = torch.IntTensor(user_click_history)
        num_user_click_history = torch.IntTensor([num_user_click_history])
        user_discrete_feature = torch.IntTensor(user_discrete_feature)
        item_id = torch.IntTensor([item_id])
        item_discrete_feature = torch.IntTensor(item_discrete_feature)
        item_cont_feature = torch.FloatTensor(item_cont_feature)
        label = torch.IntTensor([label])

        # print(num_user_click_history)

        return user_click_history, num_user_click_history, user_discrete_feature, \
               item_id, item_discrete_feature, item_cont_feature, label
```

```python id="pmE37O6yjhqc"
def load_test_data(filename):
    # item info
    df_item_info = pd.read_parquet('item_info.parquet.snappy')
    item_info_dict = {}
    for i in tqdm(range(df_item_info.shape[0])):
        item_id = df_item_info.at[i, 'item_id'] 

        item_discrete = df_item_info.at[i, 'item_vec'].split(',')[:3]
        item_cont = df_item_info.at[i, 'item_vec'].split(',')[-2:]
        price = df_item_info.at[i, 'price'] / 3000
        loc = df_item_info.at[i, 'location'] - 1 # 0~2

        item_cont.append(price) # 2 + 1
        item_discrete.append(loc) # 3 + 1

        item_cont = [float(it) for it in item_cont]
        item_discrete = [int(it) for it in item_discrete]
        item_discrete[0] = item_discrete[0] - 1 # 1~4 -> 0~3
        item_discrete[2] = item_discrete[2] - 1 # 1~2 -> 0~1

        item_info_dict[int(item_id)] = {
            'cont': np.array(item_cont, dtype=np.float64),
            'discrete': np.array(item_discrete, dtype=np.int64),
        }

    # testset
    test_samples = []
    df_test = pd.read_parquet('{}.parquet.snappy'.format(filename))

    # shuffle
    total_num = int(df_test.shape[0])

    for i in tqdm(range(total_num)):
        if df_test.at[i, 'user_click_history'] == '0:0':
            user_click_list = [0]
        else:
            user_click_list = df_test.at[i, 'user_click_history'].split(',')
            user_click_list = [int(sample.split(':')[0]) for sample in user_click_list]
        num_user_click_history = len(user_click_list)
        tmp = np.zeros(400, dtype=np.int64)
        tmp[:len(user_click_list)] = user_click_list
        user_click_list = tmp
        
        exposed_items = [int(s) for s in df_test.at[i, 'exposed_items'].split(',')]
        labels = [int(s) for s in df_test.at[i, 'labels'].split(',')]

        user_portrait = [int(s) for s in df_test.at[i, 'user_protrait'].split(',')]
        # portraitidx_to_idx_dict_list: list of 10 dict, int:int
        for j in range(10):
            user_portrait[j] = portraitidx_to_idx_dict_list[j][user_portrait[j]]
        for k in range(9):
            one_sample = {
                'user_click_list': user_click_list,
                'num_user_click_history': num_user_click_history,
                'user_portrait': np.array(user_portrait, dtype=np.int64),
                'item_id': exposed_items[k],
            }
            test_samples.append(one_sample)
    return item_info_dict, test_samples
```

```python id="gujtFyHPj71n"
class BigDataCupTestDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 item_info_dict,
                 database
                ):
        super().__init__()
        self.item_info_dict = item_info_dict
        self.database = database

    def __len__(self, ):
        return len(self.database)

    def __getitem__(self, idx):
        one_sample = self.database[idx]
        user_click_history = one_sample['user_click_list']
        num_user_click_history = one_sample['num_user_click_history']
        user_discrete_feature = one_sample['user_portrait']
        item_id = one_sample['item_id']
        item_discrete_feature = self.item_info_dict[item_id]['discrete']
        item_cont_feature = self.item_info_dict[item_id]['cont']

        user_click_history = torch.IntTensor(user_click_history)
        num_user_click_history = torch.IntTensor([num_user_click_history])
        user_discrete_feature = torch.IntTensor(user_discrete_feature)
        item_id = torch.IntTensor([item_id])
        item_discrete_feature = torch.IntTensor(item_discrete_feature)
        item_cont_feature = torch.FloatTensor(item_cont_feature)

        return user_click_history, num_user_click_history, user_discrete_feature, \
               item_id, item_discrete_feature, item_cont_feature
```

```python colab={"base_uri": "https://localhost:8080/"} id="K1UNzz7okGPo" executionInfo={"status": "ok", "timestamp": 1637147035235, "user_tz": -330, "elapsed": 29732, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0f32c68d-be42-4b93-98e5-25b307e0f39f"
item_info_dict, train_samples, val_samples = load_data()

train_ds = BigDataCupDataset(item_info_dict, train_samples)
train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=32, shuffle=True)

val_ds = BigDataCupDataset(item_info_dict, val_samples)
val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=9, shuffle=False)
```

<!-- #region id="H-OwiaO5j9pN" -->
## Training
<!-- #endregion -->

```python id="FABFoAHdkRZc"
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc


## below: only applicable for batch_size==9 in validation
def real_acc_calc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    if acc != 1.0:
        return 0
    else:
        return 1

def real_acc_rule_calc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    
    cum_sum = 0
    for j in range(9):
        if y_pred_tag[j][0] == 1:
            cum_sum += 1
        if j == 2 and cum_sum != 3:
            y_pred_tag[3:] = 0
            break
        if j == 5 and cum_sum != 6:
            y_pred_tag[6:] = 0
            break
    
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    if acc != 1.0:
        return 0
    else:
        return 1

def real_acc_rule2_calc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    
    cum_sum = 0
    for j in range(9):
        k = 8 - j
        if k >= 6 and y_pred_tag[k][0] == 1:
            y_pred_tag[:6] = 1
        if k >= 3 and y_pred_tag[k][0] == 1:
            y_pred_tag[:3] = 1
    
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    if acc != 1.0:
        return 0
    else:
        return 1
```

```python colab={"base_uri": "https://localhost:8080/"} id="DcET8OxzkXJn" executionInfo={"status": "ok", "timestamp": 1637153479299, "user_tz": -330, "elapsed": 6396561, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ac94c6f2-20a0-4fa3-d42b-0669c7d5c2b8"
model = VanillaBaseModel(
    num_items=381,
    dim_item_emb=16,
    dim_item_discrete_feature_emb=16,
    dim_user_discrete_feature_emb=16,
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


NUM_EPOCH = 2
for epoch_idx in range(NUM_EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    train_cnt = 0
    train_acc_sum = 0
    
    for i, data in enumerate(train_dl, 0):
        model.train()

        # get the inputs; data is a list of [inputs, labels]
        user_click_history, num_user_click_history, user_discrete_feature, \
               item_id, item_discrete_feature, item_cont_feature, label = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(user_click_history, num_user_click_history, user_discrete_feature, \
               item_id, item_discrete_feature, item_cont_feature)

        loss = criterion(outputs, label.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        acc = binary_acc(outputs, label)
        train_acc_sum += acc
        train_cnt += 1

        if i % 1000 == 1:    # print every 2000 mini-batches
            print('----- TRAIN -----')
            print('[%d, %5d] loss: %.3f' % (epoch_idx + 1, i + 1, running_loss))
            print('- acc:', train_acc_sum / train_cnt, flush=True)

            running_loss = 0.0
            train_cnt = 0
            train_acc_sum = 0
            # print(outputs, label)

            ## val
            model.eval()
            cnt = 0
            acc_sum = 0
            real_acc_sum = 0
            real_rule_acc_sum = 0
            real_rule2_acc_sum = 0
            
            for _, val_data in tqdm(enumerate(val_dl, 0)):
                user_click_history, num_user_click_history, user_discrete_feature, \
                    item_id, item_discrete_feature, item_cont_feature, label = val_data
                outputs = model(user_click_history, num_user_click_history, user_discrete_feature, \
                    item_id, item_discrete_feature, item_cont_feature)
                acc = binary_acc(outputs, label)
                real_acc = real_acc_calc(outputs, label)
                real_rule_acc = real_acc_rule_calc(outputs, label)
                real_rule2_acc = real_acc_rule2_calc(outputs, label)
                
                acc_sum += acc
                real_acc_sum += real_acc
                real_rule_acc_sum += real_rule_acc
                real_rule2_acc_sum += real_rule2_acc

                cnt += 1
            print('----- VAL -----')
            print('- acc:', acc_sum / cnt)
            print('- real acc:', real_acc_sum / cnt)
            print('- real rule acc:', real_rule_acc_sum / cnt)
            print('- real rule2 acc:', real_rule2_acc_sum / cnt)

print('Finished Training')
```

```python id="jC1RcgRRko37"
torch.save(model, 'vanilla_model_epoch1.2.pth')
torch.save(model, 'vanilla_layernorm_model_epoch1.2.pth')
```

<!-- #region id="sXDy2zc1kqG1" -->
## Testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="U0_NHXBbkzW1" executionInfo={"status": "ok", "timestamp": 1637154841088, "user_tz": -330, "elapsed": 23486, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8455bc52-8051-41c6-aaed-c9ad404b5917"
item_info_dict, test_samples = load_test_data('track1_testset')

test_ds = BigDataCupTestDataset(item_info_dict, test_samples)
test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=9, shuffle=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="gzAVd-w8k9Cz" executionInfo={"status": "ok", "timestamp": 1637155214553, "user_tz": -330, "elapsed": 369996, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="51ae68f7-1891-4064-9b89-4b06d73169a6"
model = model.eval()

fp = open('vanilla_model_epoch1.2_track1.csv', 'w')
print('id,category', file=fp)

for i, data in tqdm(enumerate(test_dl, 0)):
    user_click_history, num_user_click_history, user_discrete_feature, \
            item_id, item_discrete_feature, item_cont_feature = data

    # forward + backward + optimize
    outputs = model(user_click_history, num_user_click_history, user_discrete_feature, \
            item_id, item_discrete_feature, item_cont_feature)
    
    y_pred_tag = torch.round(torch.sigmoid(outputs))

    ## rule1
    # cum_sum = 0
    # for j in range(9):
    #     if y_pred_tag[j][0] == 1:
    #         cum_sum += 1
    #     if j == 2 and cum_sum != 3:
    #         y_pred_tag[3:] = 0
    #         break
    #     if j == 5 and cum_sum != 6:
    #         y_pred_tag[6:] = 0
    #         break

    ## rule2
    cum_sum = 0
    for j in range(9):
        k = 8 - j
        if k >= 6 and y_pred_tag[k][0] == 1:
            y_pred_tag[:6] = 1
        if k >= 3 and y_pred_tag[k][0] == 1:
            y_pred_tag[:3] = 1
    
    y_pred_tag = list(y_pred_tag.detach().numpy()[:, 0].astype(np.int32))
    y_pred_tag = [str(a) for a in y_pred_tag]
    p = ' '.join(y_pred_tag)
    print(f'{i+1},{p}', file=fp)
    # break

fp.close()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="rkfZifS6EPNp" executionInfo={"status": "ok", "timestamp": 1637155417890, "user_tz": -330, "elapsed": 697, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="78c6ded5-c741-4edb-f681-26f87aafed53"
pd.read_csv('/content/vanilla_model_epoch1.2_track1.csv')
```
