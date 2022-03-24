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

<!-- #region id="PhA55DFXOfwR" -->
# IEEE Challenge 2021 Track1 Session-aware Recommendation with Transformer Session Prediction
<!-- #endregion -->

<!-- #region id="a_JhGn3SqC-a" -->
We will implement a model to predict which session the user has purchased.
<!-- #endregion -->

<!-- #region id="l_QXu0x4qRRt" -->
Input：
1. 9 products displayed to users (emb, discrete attributes, continuous attributes of products)
3. User's click history (emb, discrete attributes, continuous attributes of products)
3. User attributes (discrete attributes)

Output:
1. The user purchased in the session (purchased 0-3, 4-6, 7-9, three types of sessions)
2. Whether the user bought these 9 products (you can use the 4 types of product reweighting loss mentioned by Gaochen)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="joAvJDL8hIie" executionInfo={"status": "ok", "timestamp": 1637146618868, "user_tz": -330, "elapsed": 8500, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9423712b-58be-4eb9-ec86-bb78fd7761fd"
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/train.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/item_info.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track1_testset.parquet.snappy
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track2_testset.parquet.snappy
```

```python colab={"base_uri": "https://localhost:8080/"} id="597GI-kjCYuj" outputId="d7cd274c-98b4-435a-a2fe-35014f6224ed"
!pip install einops
```

```python id="K4cozZgaq-CJ"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle
```

```python id="Xe2QKPaICaa-"
import einops
```

<!-- #region id="iI3v-E1gq3HQ" -->
## model
<!-- #endregion -->

```python id="rUIJbrWoq_Yc"
class SessionPredictionModel(nn.Module):
    def __init__(self, 
                 num_items,
                 dim_item_emb=64,
                 dim_item_discrete_feature_emb=16,
                 dim_user_discrete_feature_emb=16,
                ):
        super().__init__()
        self.num_items = num_items
        self.NUM_ITEM_DISCRETE_FEATURE = 3+1 # item_vec3 + location1
        self.NUM_ITEM_CONT_FEATURE = 2+1 # item_vec2 + price1
        self.NUM_USER_DISCRETE_FEATURE = 10
        self.dim_item_emb = dim_item_emb
        self.dim_item_discrete_feature_emb = dim_item_discrete_feature_emb
        self.dim_user_discrete_feature_emb = dim_user_discrete_feature_emb

        # item emb
        self.item_emb = nn.Embedding(self.num_items + 1, self.dim_item_emb) # num_items + 1

        # item discrete feature
        self.item_discrete_feature_emb_list = nn.ModuleList()
        num_unique_value_list = [4, 10, 2, 3]
        for i in range(self.NUM_ITEM_DISCRETE_FEATURE):
            num_unique_value = num_unique_value_list[i]
            self.item_discrete_feature_emb_list.append(
                nn.Embedding(num_unique_value, self.dim_item_discrete_feature_emb)
            )
        
        # user discrete feature
        self.user_discrete_feature_emb_list = nn.ModuleList()
        num_unique_value_list = [3, 1430, 20, 10, 198, 52, 3, 13, 2, 2347]
        for i in range(self.NUM_USER_DISCRETE_FEATURE):
            num_unique_value = num_unique_value_list[i]
            self.user_discrete_feature_emb_list.append(
                nn.Embedding(num_unique_value, self.dim_user_discrete_feature_emb)
            )

        # backbone
        self.backbone = nn.Sequential(
            nn.Linear(
                # user_click_history
                self.dim_item_emb + self.NUM_ITEM_DISCRETE_FEATURE * self.dim_item_discrete_feature_emb + self.NUM_ITEM_CONT_FEATURE + 
                # nine items
                9 * (self.dim_item_emb + self.NUM_ITEM_DISCRETE_FEATURE * self.dim_item_discrete_feature_emb + self.NUM_ITEM_CONT_FEATURE) +
                # user
                self.NUM_USER_DISCRETE_FEATURE * self.dim_user_discrete_feature_emb
                , 400), 
            nn.PReLU(),
            nn.Linear(400, 200),
            nn.PReLU()
        )

        # session prediction head
        self.session_prediction_head = nn.Sequential(
            nn.Linear(200, 80),
            nn.PReLU(),
            nn.Linear(80, 20),
            nn.PReLU(),
            nn.Linear(20, 4)
        )

        # buy prediction head
        self.buy_prediction_head = nn.Sequential(
            nn.Linear(200, 80),
            nn.PReLU(),
            nn.Linear(80, 20),
            nn.PReLU(),
            nn.Linear(20, 9)
        )

    def get_item_emb_attr(self, 
                          item_id, 
                          item_cont_feature, 
                          item_discrete_feature):
        """
        param:
            item_id:                [N, 9]
            item_cont_feature:     [N, 9, NUM_ITEM_CONT_FEATURE]
            item_discrete_feature: [N, 9, NUM_USER_DISCRETE_FEATURE]
        
        return: 
            emb_attr:
                [N, 9, dim_item_emb
                       + NUM_ITEM_CONT_FEATURE
                       + NUM_USER_DISCRETE_FEATURE * dim_user_discrete_feature_emb
                ]

        note: 
            above, 9 can be an arbitrary number, e.g. 400
        """
        
        # item emb
        item_emb = self.item_emb(item_id) # [N, 9, dim_item_emb]
        
        # item discrete feature emb
        tmp = []
        for i in range(self.NUM_ITEM_DISCRETE_FEATURE):
            # print(i)
            # print(item_discrete_feature[:, i])
            tmp.append(
                self.item_discrete_feature_emb_list[i](item_discrete_feature[:, :, i]) # [N, 9, dim_user_discrete_feature_emb]
            )
        item_discrete_feature_emb = torch.cat(tmp, dim=-1)

        # concat [N, 9, D]
        emb_attr = torch.cat([
             item_emb, 
             item_discrete_feature_emb, 
             item_cont_feature
        ], dim=-1)

        return emb_attr

    def forward(self,
                user_click_history, user_click_history_discrete_feature, user_click_history_cont_feature, num_user_click_history,
                item_id, item_discrete_feature, item_cont_feature,
                user_discrete_feature,
                ):
        """
            user_click_history: [N, 400]
            user_click_history_discrete_feature: [N, 400, 3+1]
            user_click_history_cont_feature: [N, 400, 2+1]
            num_user_click_history: [N, 1]

            item_id: [N, 9]
            item_discrete_feature: [N, 9, 3+1] item_vec3 + location1
            item_cont_feature: [N, 9, 2+1] item_vec2 + price1

            user_discrete_feature: [N, 10]
        """

        batch_size = user_click_history.size()[0]
        
        tmp = self.get_item_emb_attr(user_click_history, 
                                     user_click_history_cont_feature, 
                                     user_click_history_discrete_feature) # [N, 400, D]
        user_click_history_emb = torch.zeros( # [N, D]
            (batch_size, self.dim_item_emb
                         + self.NUM_ITEM_CONT_FEATURE
                         + self.NUM_ITEM_DISCRETE_FEATURE * self.dim_item_discrete_feature_emb)
        )
        for i in range(batch_size):
            aa = tmp[i, :num_user_click_history[i], :] # [N, 400, D] -> [400-, D]
            a = torch.mean(aa, dim=0) # [400-, D] -> [D]
            user_click_history_emb[i] = a

        nine_item_emb = self.get_item_emb_attr(item_id, 
                                               item_cont_feature, 
                                               item_discrete_feature) # [N, 9, D]
        nine_item_emb = einops.rearrange(nine_item_emb, 'N B D -> N (B D)') # [N, 9D]

        tmp = []
        for i in range(self.NUM_USER_DISCRETE_FEATURE):
            tmp.append(
                self.user_discrete_feature_emb_list[i](user_discrete_feature[:, i]) # [N, dim_user_discrete_feature_emb]
            )
        user_discrete_feature_emb = torch.cat(tmp, dim=1)

        all_emb = torch.cat([user_click_history_emb, 
                             nine_item_emb,
                             user_discrete_feature_emb,
                            ], dim=1) # [N, D]
        
        feat = self.backbone(all_emb) # [N, 1]
        session_pred = self.session_prediction_head(feat)
        buy_pred = self.buy_prediction_head(feat)
        return session_pred, buy_pred # [N, 4], [N, 9]
```

```python colab={"base_uri": "https://localhost:8080/"} id="0zRuRnzOr40p" outputId="ad152d10-7983-40fb-bf2b-4718ad746df7"
m = SessionPredictionModel(num_items=381,
                           dim_item_emb=64,
                           dim_item_discrete_feature_emb=16,
                           dim_user_discrete_feature_emb=16
                           )

B = 7
m(
    user_click_history=torch.ones([B, 400], dtype=torch.int32),
    user_click_history_discrete_feature=torch.ones([B, 400, 4], dtype=torch.int32),
    user_click_history_cont_feature=torch.randn([B, 400, 3]),
    num_user_click_history=torch.ones([B, 1], dtype=torch.int32) * 10,
    user_discrete_feature=torch.ones([B, 10], dtype=torch.int32),
    item_id=torch.ones([B, 9], dtype=torch.int32),
    item_discrete_feature=torch.ones([B, 9, 4], dtype=torch.int32),
    item_cont_feature=torch.randn([B, 9, 3]),
)
```

<!-- #region id="XLUSRSm_q0m6" -->
## data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eZO24jMGrxur" outputId="24047be7-1262-468b-8396-97a60a7f41d5"
## user portrait
data_path='/content'
# portraitidx_to_idx_dict_list: list of 10 dict, int:int

portraitidx_to_idx_dict_list = []
for i in range(10):
    portraitidx_to_idx_dict_list.append(dict())
acculumated_idx = [0] * 10


df_train = pd.read_parquet(f'{data_path}/trainset.parquet.snappy')
for i in tqdm(range(df_train.shape[0])):
    user_portrait = [int(s) for s in df_train.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
            acculumated_idx[idx] += 1

df_test1 = pd.read_parquet(f'{data_path}/track1_testset.parquet.snappy')
for i in tqdm(range(df_test1.shape[0])):
    user_portrait = [int(s) for s in df_test1.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
            acculumated_idx[idx] += 1

df_test2 = pd.read_parquet(f'{data_path}/track2_testset.parquet.snappy')
for i in tqdm(range(df_test2.shape[0])):
    user_portrait = [int(s) for s in df_test2.at[i, 'user_protrait'].split(',')]
    for idx, u in enumerate(user_portrait):
        if portraitidx_to_idx_dict_list[idx].get(u, -1) == -1:
            portraitidx_to_idx_dict_list[idx][u] = acculumated_idx[idx]
            acculumated_idx[idx] += 1

acculumated_idx
```

```python id="yGmk9gZgqjQJ"
def load_item_info(data_path='/content'):
    # item info
    df_item_info = pd.read_parquet(f'{data_path}/item_info.parquet.snappy')
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
    return item_info_dict
```

```python colab={"base_uri": "https://localhost:8080/"} id="gbjFZ4SPAJRI" outputId="a13bd2d8-4f60-41e3-c841-82052acd2e34"
item_info_dict = load_item_info(data_path='/content')
```

```python id="WvWsdWAP1t00"
def load_train_data(data_path='/content'):
    # trainset
    train_samples = []
    val_samples = []
    df_train = pd.read_parquet(f'{data_path}/trainset.parquet.snappy')

    # shuffle
    df_train = shuffle(df_train, random_state=2333).reset_index()
    total_num = int(df_train.shape[0])
    num_train = int(total_num * 0.95)
    num_val = total_num - num_train # 5% validation data

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
        one_sample = {
            'user_click_list': user_click_list,
            'num_user_click_history': num_user_click_history,
            'user_portrait': np.array(user_portrait, dtype=np.int64),
            'item_id': np.array(exposed_items, dtype=np.int64),
            'label': np.array(labels, dtype=np.int64)
        }
        if i < num_train:
            train_samples.append(one_sample)
        else:
            val_samples.append(one_sample)
    return train_samples, val_samples
```

```python colab={"base_uri": "https://localhost:8080/"} id="GaHYRMrE4oV5" outputId="1b62875c-3893-4232-eb50-bdfcb5baaaff"
train_samples, val_samples = load_train_data(data_path='/content')
```

```python id="LhnPDrwEq8GK"
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
        label = one_sample['label']

        user_click_history_discrete_feature = np.zeros((400, 3+1)).astype(np.int64)
        user_click_history_cont_feature = np.zeros((400, 2+1)).astype(np.float64)
        for i in range(num_user_click_history):
            if user_click_history[i] == 0:
                continue
            else:
                user_click_history_discrete_feature[i] = self.item_info_dict[user_click_history[i]]['discrete']
                user_click_history_cont_feature[i] = self.item_info_dict[user_click_history[i]]['cont']

        item_discrete_feature = np.zeros((9, 3+1)).astype(np.int64)
        item_cont_feature = np.zeros((9, 2+1)).astype(np.float64)
        for i in range(9):
            item_discrete_feature[i] = self.item_info_dict[item_id[i]]['discrete']
            item_cont_feature[i] = self.item_info_dict[item_id[i]]['cont']

        session_label = 0 # 0,1,2,3
        for i in range(9):
            if label[i]:
                session_label = 1
            if i >= 3 and label[i]:
                session_label = 2
            if i >= 6 and label[i]:
                session_label = 3

        user_click_history = torch.IntTensor(user_click_history)
        user_click_history_discrete_feature = torch.IntTensor(user_click_history_discrete_feature)
        user_click_history_cont_feature = torch.FloatTensor(user_click_history_cont_feature)
        num_user_click_history = torch.IntTensor([num_user_click_history])
        user_discrete_feature = torch.IntTensor(user_discrete_feature)
        item_id = torch.IntTensor(item_id)
        item_discrete_feature = torch.IntTensor(item_discrete_feature)
        item_cont_feature = torch.FloatTensor(item_cont_feature)
        label = torch.IntTensor(label)
        session_label = session_label
        
        return user_click_history, \
            user_click_history_discrete_feature, user_click_history_cont_feature, \
            num_user_click_history, \
            item_id, item_discrete_feature, item_cont_feature, \
            user_discrete_feature, label, session_label
```

```python id="GeMu5TTrsW4y"
ds = BigDataCupDataset(item_info_dict, train_samples)
```

```python colab={"base_uri": "https://localhost:8080/"} id="O4gRymk3y4NA" outputId="635db1ee-807f-4b42-e50f-a5a02561b531"
train_samples, val_samples = load_train_data()

train_ds = BigDataCupDataset(item_info_dict, train_samples)
train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=32, shuffle=True)

val_ds = BigDataCupDataset(item_info_dict, val_samples)
val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=32, shuffle=False)
```

<!-- #region id="TLla7fQWACja" -->
## train
<!-- #endregion -->

```python id="SgJASuk0Zh4y"
def binary_acc(sess_pred, y_pred, y_test):
    # print(sess_pred)
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    y_pred_tag_intact = y_pred_tag.clone()


    ##################################
    ## vanilla
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc1 = correct_results_sum / y_test.shape[0] / 9

    real_acc1 = 0.0
    for i in range(y_test.shape[0]):
        correct_results_sum = (y_pred_tag[i] == y_test[i]).sum().float()
        one_acc = correct_results_sum / 9
        if one_acc == 1:
            real_acc1 += 1
    real_acc1 = real_acc1 / y_test.shape[0]
    # print(y_pred_tag)

    ####################################
    ## use sess to refine y_pred_tag
    for i in range(y_test.shape[0]):
        if sess_pred[i] == 0:
            y_pred_tag[i][:] = 0
        elif sess_pred[i] == 1:
            y_pred_tag[i][3:] = 0
        elif sess_pred[i] == 2:
            y_pred_tag[i][:3] = 1
            y_pred_tag[i][6:] = 0
        elif sess_pred[i] == 3:
            y_pred_tag[i][:6] = 1

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc2 = correct_results_sum / y_test.shape[0] / 9

    real_acc2 = 0.0
    for i in range(y_test.shape[0]):
        correct_results_sum = (y_pred_tag[i] == y_test[i]).sum().float()
        one_acc = correct_results_sum / 9
        if one_acc == 1:
            real_acc2 += 1
    real_acc2 = real_acc2 / y_test.shape[0]
    # print(y_pred_tag)

    #######################################
    ## rule 2
    y_pred_tag = y_pred_tag_intact
    acc_rule2 = 0.0
    real_acc_rule2 = 0.0
    for i in range(y_test.shape[0]):
        for j in range(9):
            k = 8 - j
            if k >= 6 and y_pred_tag[i][k] == 1:
                y_pred_tag[i][:6] = 1
            if k >= 3 and y_pred_tag[i][k] == 1:
                y_pred_tag[i][:3] = 1
        
        correct_results_sum = (y_pred_tag[i] == y_test[i]).sum().float()
        a = correct_results_sum / 9
        acc_rule2 += a
        if a == 1:
            real_acc_rule2 += 1
    acc_rule2 = acc_rule2 / y_test.shape[0]
    real_acc_rule2 = real_acc_rule2 / y_test.shape[0]
    # print(y_pred_tag)


    return acc1, acc2, acc_rule2, real_acc1, real_acc2, real_acc_rule2
```

```python id="RK0VlCA3hmmz"
model = SessionPredictionModel(num_items=381,
    dim_item_emb=32,
    dim_item_discrete_feature_emb=8,
    dim_user_discrete_feature_emb=8
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="CLvz60iaAOLo" outputId="66013af6-8301-4a77-fdc5-f2c7a8e42908"
model
```

```python id="N3eOxzVAHhr9"
sess_criterion = nn.CrossEntropyLoss()
buy_criterion = nn.BCEWithLogitsLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

```python colab={"base_uri": "https://localhost:8080/"} id="5uh3Bd3EASid" outputId="75a49445-90aa-4bee-c4e8-566eadaea4e7"
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

NUM_EPOCH = 2
for epoch_idx in range(NUM_EPOCH):  # loop over the dataset multiple times

    running_sess_loss = 0.0
    running_buy_loss = 0.0
    train_cnt = 0
    train_sess_acc_sum = 0
    train_buy_acc1_sum = 0
    train_buy_real_acc1_sum = 0
    train_buy_acc2_sum = 0
    train_buy_real_acc2_sum = 0
    train_buy_acc_rule2_sum = 0
    train_buy_real_acc_rule2_sum = 0
    train_cnt_session_0 = train_cnt_session_1 = train_cnt_session_2 = train_cnt_session_3 = 0
    
    for i, data in enumerate(train_dl, 0):
        model.train()

        # get the inputs; data is a list of [inputs, labels]
        user_click_history, \
            user_click_history_discrete_feature, user_click_history_cont_feature, \
            num_user_click_history, \
            item_id, item_discrete_feature, item_cont_feature, \
            user_discrete_feature, label, session_label = data

        train_batch_size = user_click_history.shape[0]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        sess_outputs, buy_outputs = model(
            user_click_history,
            user_click_history_discrete_feature,
            user_click_history_cont_feature,
            num_user_click_history,
            item_id,
            item_discrete_feature,
            item_cont_feature,
            user_discrete_feature
        )

        sess_loss = sess_criterion(sess_outputs, session_label)
        buy_loss = buy_criterion(buy_outputs, label.float())
        loss = sess_loss + buy_loss

        loss.backward()
        optimizer.step()

        # print statistics
        running_sess_loss += sess_loss.item()
        running_buy_loss += buy_loss.item()

        _, sess_predicted = torch.max(sess_outputs.data, 1)
        sess_acc = (sess_predicted == session_label).sum().item() / train_batch_size
        # buy_acc1, buy_acc2, buy_real_acc1, buy_real_acc2 = binary_acc(sess_predicted, buy_outputs, label)
        buy_acc1, buy_acc2, buy_acc_rule2, buy_real_acc1, buy_real_acc2, buy_real_acc_rule2 = binary_acc(sess_predicted, buy_outputs, label)

        train_sess_acc_sum += sess_acc
        train_buy_acc1_sum += buy_acc1
        train_buy_real_acc1_sum += buy_real_acc1
        train_buy_acc2_sum += buy_acc2
        train_buy_real_acc2_sum += buy_real_acc2
        train_buy_acc_rule2_sum += buy_acc_rule2
        train_buy_real_acc_rule2_sum += buy_real_acc_rule2
        train_cnt += 1

        train_cnt_session_0 += torch.sum(session_label == 0)
        train_cnt_session_1 += torch.sum(session_label == 1)
        train_cnt_session_2 += torch.sum(session_label == 2)
        train_cnt_session_3 += torch.sum(session_label == 3)

        if i % 50 == 1:
            print(i, )

        if i % 500 == 1:    # print every 2000 mini-batches
            print('----- TRAIN -----')
            print('[%d, %5d] sess loss: %.3f' % (epoch_idx + 1, i + 1, running_sess_loss / train_cnt))
            print('[%d, %5d] buy loss: %.3f'  % (epoch_idx + 1, i + 1, running_buy_loss / train_cnt))
            print('- sess acc:', train_sess_acc_sum / train_cnt, flush=True)
            print('- buy acc1:', train_buy_acc1_sum / train_cnt, flush=True)
            print('- buy real acc1:', train_buy_real_acc1_sum / train_cnt, flush=True)
            print('- buy acc2:', train_buy_acc2_sum / train_cnt, flush=True)
            print('- buy real acc2:', train_buy_real_acc2_sum / train_cnt, flush=True)
            print('- buy acc rule2:', train_buy_acc_rule2_sum / train_cnt, flush=True)
            print('- buy real acc rule2:', train_buy_real_acc_rule2_sum / train_cnt, flush=True)
            print('- train sess cnt:', train_cnt_session_0, train_cnt_session_1, train_cnt_session_2, train_cnt_session_3)

            running_sess_loss = 0.0
            running_buy_loss = 0.0
            train_cnt = 0
            train_sess_acc_sum = 0
            train_buy_acc1_sum = 0
            train_buy_real_acc1_sum = 0
            train_buy_acc2_sum = 0
            train_buy_real_acc2_sum = 0
            train_buy_acc_rule2_sum = 0
            train_buy_real_acc_rule2_sum = 0
            train_cnt_session_0 = train_cnt_session_1 = train_cnt_session_2 = train_cnt_session_3 = 0


            ## val
            model.eval()
            valid_cnt = 0
            valid_sess_acc_sum = 0
            valid_buy_acc1_sum = 0
            valid_buy_real_acc1_sum = 0
            valid_buy_acc2_sum = 0
            valid_buy_real_acc2_sum = 0
            valid_buy_acc_rule2_sum = 0
            valid_buy_real_acc_rule2_sum = 0
            
            valid_cnt_session_0 = valid_cnt_session_1 = valid_cnt_session_2 = valid_cnt_session_3 = 0

            for _, val_data in tqdm(enumerate(val_dl, 0)):
                user_click_history, \
                    user_click_history_discrete_feature, user_click_history_cont_feature, \
                    num_user_click_history, \
                    item_id, item_discrete_feature, item_cont_feature, \
                    user_discrete_feature, label, session_label = val_data
                sess_outputs, buy_outputs = model(
                    user_click_history,
                    user_click_history_discrete_feature,
                    user_click_history_cont_feature,
                    num_user_click_history,
                    item_id,
                    item_discrete_feature,
                    item_cont_feature,
                    user_discrete_feature
                )

                valid_batch_size = user_click_history.shape[0]

                _, sess_predicted = torch.max(sess_outputs.data, 1)
                sess_acc = (sess_predicted == session_label).sum().item() / valid_batch_size
                buy_acc1, buy_acc2, buy_acc_rule2, buy_real_acc1, buy_real_acc2, buy_real_acc_rule2 = binary_acc(sess_predicted, buy_outputs, label)

                valid_sess_acc_sum += sess_acc
                valid_buy_acc1_sum += buy_acc1
                valid_buy_real_acc1_sum += buy_real_acc1
                valid_buy_acc2_sum += buy_acc2
                valid_buy_real_acc2_sum += buy_real_acc2
                valid_buy_acc_rule2_sum += buy_acc_rule2
                valid_buy_real_acc_rule2_sum += buy_real_acc_rule2
                valid_cnt += 1

                valid_cnt_session_0 += torch.sum(session_label == 0)
                valid_cnt_session_1 += torch.sum(session_label == 1)
                valid_cnt_session_2 += torch.sum(session_label == 2)
                valid_cnt_session_3 += torch.sum(session_label == 3)

            print('----- VAL -----')
            print('- sess acc:', valid_sess_acc_sum / valid_cnt)
            print('- buy acc1:', valid_buy_acc1_sum / valid_cnt)
            print('- buy real acc1:', valid_buy_real_acc1_sum / valid_cnt)
            print('- buy acc2:', valid_buy_acc2_sum / valid_cnt)
            print('- buy real acc2:', valid_buy_real_acc2_sum / valid_cnt)
            print('- buy acc rule2:', valid_buy_acc_rule2_sum / valid_cnt)
            print('- buy real acc rule2:', valid_buy_real_acc_rule2_sum / valid_cnt)
            print('valid sess cnt:', valid_cnt_session_0, valid_cnt_session_1, valid_cnt_session_2, valid_cnt_session_3)

print('Finished Training')
```

```python id="0Df1QMAUl0tJ"
torch.save(model, '4sess_pred_deepermodel_epoch2.pth')
```

<!-- #region id="fs0871p6l08l" -->
## test
<!-- #endregion -->

```python id="z2e8vb-bC0eE"
def load_test_data(data_path='/content',
                   filename='track1_testset.parquet.snappy'):
    test_samples = []
    df_test = pd.read_parquet(f'{data_path}/{filename}')

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

        user_portrait = [int(s) for s in df_test.at[i, 'user_protrait'].split(',')]
        # portraitidx_to_idx_dict_list: list of 10 dict, int:int
        for j in range(10):
            user_portrait[j] = portraitidx_to_idx_dict_list[j][user_portrait[j]]
        one_sample = {
            'user_click_list': user_click_list,
            'num_user_click_history': num_user_click_history,
            'user_portrait': np.array(user_portrait, dtype=np.int64),
            'item_id': np.array(exposed_items, dtype=np.int64),
        }
        test_samples.append(one_sample)
    return test_samples



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

        user_click_history_discrete_feature = np.zeros((400, 3+1)).astype(np.int64)
        user_click_history_cont_feature = np.zeros((400, 2+1)).astype(np.float64)
        for i in range(num_user_click_history):
            if user_click_history[i] == 0:
                continue
            else:
                user_click_history_discrete_feature[i] = self.item_info_dict[user_click_history[i]]['discrete']
                user_click_history_cont_feature[i] = self.item_info_dict[user_click_history[i]]['cont']

        item_discrete_feature = np.zeros((9, 3+1)).astype(np.int64)
        item_cont_feature = np.zeros((9, 2+1)).astype(np.float64)
        for i in range(9):
            item_discrete_feature[i] = self.item_info_dict[item_id[i]]['discrete']
            item_cont_feature[i] = self.item_info_dict[item_id[i]]['cont']

        user_click_history = torch.IntTensor(user_click_history)
        user_click_history_discrete_feature = torch.IntTensor(user_click_history_discrete_feature)
        user_click_history_cont_feature = torch.FloatTensor(user_click_history_cont_feature)
        num_user_click_history = torch.IntTensor([num_user_click_history])
        user_discrete_feature = torch.IntTensor(user_discrete_feature)
        item_id = torch.IntTensor(item_id)
        item_discrete_feature = torch.IntTensor(item_discrete_feature)
        item_cont_feature = torch.FloatTensor(item_cont_feature)
        
        return user_click_history, \
            user_click_history_discrete_feature, user_click_history_cont_feature, \
            num_user_click_history, \
            item_id, item_discrete_feature, item_cont_feature, \
            user_discrete_feature

```

```python colab={"base_uri": "https://localhost:8080/"} id="xBTlmys4XqKF" outputId="83819fb5-7325-47af-f108-950fafce3a1e"
test_samples = load_test_data(data_path='/content', 
                                              filename='track1_testset.parquet.snappy')

test_ds = BigDataCupTestDataset(item_info_dict, test_samples)
test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=32, shuffle=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="fbDGC0xUl7RX" outputId="6fd8d63b-347c-4a91-b6cd-314282b017ec"
model = model.eval()

fp = open('4sess_pred_model_epoch2_track1.parquet.snappy', 'w')
print('id,category', file=fp)

bs = 32

for i, data in tqdm(enumerate(test_dl, 0)):
    user_click_history, \
        user_click_history_discrete_feature, user_click_history_cont_feature, \
        num_user_click_history, \
        item_id, item_discrete_feature, item_cont_feature, \
        user_discrete_feature = data
    sess_outputs, buy_outputs = model(
        user_click_history,
        user_click_history_discrete_feature,
        user_click_history_cont_feature,
        num_user_click_history,
        item_id,
        item_discrete_feature,
        item_cont_feature,
        user_discrete_feature
    )

    y_pred_tag = torch.round(torch.sigmoid(buy_outputs))
    _, sess_pred = torch.max(sess_outputs.data, 1)
        
    for j in range(y_pred_tag.shape[0]):
        if sess_pred[j] == 0:
            y_pred_tag[j][:] = 0
        elif sess_pred[j] == 1:
            y_pred_tag[j][3:] = 0
        elif sess_pred[j] == 2:
            y_pred_tag[j][:3] = 1
            y_pred_tag[j][6:] = 0
        elif sess_pred[j] == 3:
            y_pred_tag[j][:6] = 1

        tmp = list(y_pred_tag[j].detach().numpy().astype(np.int32))
        tmp = [str(a) for a in tmp]
        p = ' '.join(tmp)
        print(f'{i * bs + j + 1},{p}', file=fp)
    # break

fp.close()
```

<!-- #region id="48tdWQuI0UqW" -->
## validation analysis
<!-- #endregion -->

```python id="VAXYAVTKoEWp"
m = torch.load('4sess_pred_deepermodel_epoch2.pth')
```

```python colab={"base_uri": "https://localhost:8080/"} id="XZyAyicFzfNH" outputId="40a2bb52-b91d-4060-a18b-c9d2f3d299f6"
m
```

```python colab={"base_uri": "https://localhost:8080/"} id="lyGOaITH2aZe" outputId="9985c903-5527-4089-be27-f4bf2af79590"
train_samples, val_samples = load_train_data()

# train_ds = BigDataCupDataset(item_info_dict, train_samples)
# train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=32, shuffle=True)

val_ds = BigDataCupDataset(item_info_dict, val_samples)
val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=32, shuffle=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="s9rLPFfPzhmw" outputId="46ab1962-81f5-41fd-f8bc-c0e384666334"
model = m.eval()
# sess_pred_list = []
# sess_gt_list = []

one_zero  = np.zeros(9)
zero_one  = np.zeros(9)
one_one   = np.zeros(9)
zero_zero = np.zeros(9)

pred_num_list = []
gt_num_list = []

valid_cnt = 0
valid_sess_acc_sum = 0
valid_buy_acc1_sum = 0
valid_buy_real_acc1_sum = 0
valid_buy_acc2_sum = 0
valid_buy_real_acc2_sum = 0
valid_buy_acc_rule2_sum = 0
valid_buy_real_acc_rule2_sum = 0

for i, data in tqdm(enumerate(val_dl, 0)):
    user_click_history, \
        user_click_history_discrete_feature, user_click_history_cont_feature, \
        num_user_click_history, \
        item_id, item_discrete_feature, item_cont_feature, \
        user_discrete_feature, label, session_label = data
    sess_outputs, buy_outputs = model(
        user_click_history,
        user_click_history_discrete_feature,
        user_click_history_cont_feature,
        num_user_click_history,
        item_id,
        item_discrete_feature,
        item_cont_feature,
        user_discrete_feature
    )
    bs = user_click_history.shape[0]

    ## let all 0,1,2 item buy (this will reduce performance, tested)
    # buy_outputs[:, :3] = 1 

    _, sess_predicted = torch.max(sess_outputs.data, 1)
    sess_acc = (sess_predicted == session_label).sum().item() / bs
    buy_acc1, buy_acc2, buy_acc_rule2, buy_real_acc1, buy_real_acc2, buy_real_acc_rule2 = binary_acc(sess_predicted, buy_outputs, label)

    y_pred_tag = torch.round(torch.sigmoid(buy_outputs)).detach().numpy()
    label = label.numpy()

    pred_num = np.sum(y_pred_tag, axis=1)
    gt_num = np.sum(label, axis=1)
    pred_num_list.extend(list(pred_num))
    gt_num_list.extend(list(gt_num))

    valid_sess_acc_sum += sess_acc
    valid_buy_acc1_sum += buy_acc1
    valid_buy_real_acc1_sum += buy_real_acc1
    valid_buy_acc2_sum += buy_acc2
    valid_buy_real_acc2_sum += buy_real_acc2
    valid_buy_acc_rule2_sum += buy_acc_rule2
    valid_buy_real_acc_rule2_sum += buy_real_acc_rule2
    valid_cnt += 1

    for b in range(bs):
        y_pred = y_pred_tag[b]
        y_gt = label[b]
        for i in range(9):
            if y_pred[i] == 1 and y_gt[i] == 1:
                one_one[i] += 1
            elif y_pred[i] == 0 and y_gt[i] == 0:
                zero_zero[i] += 1
            elif y_pred[i] == 1 and y_gt[i] == 0:
                one_zero[i] += 1
            elif y_pred[i] == 0 and y_gt[i] == 1:
                zero_one[i] += 1

    # _, sess_pred = torch.max(sess_outputs.data, 1)
    # sess_pred_list.extend(list(sess_pred.numpy()))
    # sess_gt_list.extend(list(session_label))


print('----- VAL -----')
print('- sess acc:', valid_sess_acc_sum / valid_cnt)
print('- buy acc1:', valid_buy_acc1_sum / valid_cnt)
print('- buy real acc1:', valid_buy_real_acc1_sum / valid_cnt)
print('- buy acc2:', valid_buy_acc2_sum / valid_cnt)
print('- buy real acc2:', valid_buy_real_acc2_sum / valid_cnt)
print('- buy acc rule2:', valid_buy_acc_rule2_sum / valid_cnt)
print('- buy real acc rule2:', valid_buy_real_acc_rule2_sum / valid_cnt)


```

```python id="1B7xY2Oc2_ee"
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
```

```python colab={"base_uri": "https://localhost:8080/", "height": 541} id="rSZVhBGJ3hFR" outputId="86d245ba-1d38-4df3-9f39-b344d350083b"
# Confusion matrix whose i-th row and j-th column entry indicates 
# the number of samples with 
# true label being i-th class, and 
# predicted label being j-th class.
a = confusion_matrix(sess_gt_list, sess_pred_list)
a_per = a / np.sum(a, axis=1, keepdims=True) * 100
cm_display = ConfusionMatrixDisplay(a, display_labels=range(4)).plot(values_format='d')
cm_display = ConfusionMatrixDisplay(a_per, display_labels=range(4)).plot(values_format='2.0f')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 541} id="WLY2gryVB1aH" outputId="7639851a-3012-4f35-f67a-4d2c76e5a9bb"
a = confusion_matrix(pred_num_list, gt_num_list)
a_per = a / np.sum(a, axis=1, keepdims=True) * 100
cm_display = ConfusionMatrixDisplay(a, display_labels=range(10)).plot(values_format='d')
cm_display = ConfusionMatrixDisplay(a_per, display_labels=range(10)).plot(values_format='2.0f')
```

```python colab={"base_uri": "https://localhost:8080/"} id="C-kfakmqB3kP" outputId="006de0bf-40df-40c4-9a37-d2df52367d45"
s = 0
for i in range(10):
    s += a[i][i]
print(s)
```

```python colab={"base_uri": "https://localhost:8080/"} id="WDCD7YFJCCji" outputId="891b4968-18c1-466d-b5d6-5e676745bc13"
4605 / np.sum(a)
```

```python colab={"base_uri": "https://localhost:8080/"} id="NHtQR8-ICQBJ" outputId="3b86497a-25a9-4a37-8923-c39a659ee2b6"
np.sum(a)
```

```python colab={"base_uri": "https://localhost:8080/"} id="VPt69Xkk4XgA" outputId="2c4aaac0-2f28-498e-f9dd-c8604edbb986"
a = one_zero + zero_one + one_one + zero_zero
print(one_zero)
print(zero_one)
print(one_one)
print(zero_zero) 
print('')
print(np.round(one_zero  / a, 2))
print(np.round(zero_one  / a, 2))
print(np.round(one_one   / a, 2))
print(np.round(zero_zero / a, 2)) 
```
