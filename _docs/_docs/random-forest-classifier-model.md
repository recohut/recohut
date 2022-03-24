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

```python colab={"base_uri": "https://localhost:8080/"} id="twDQ_UNWlgdm" executionInfo={"status": "ok", "timestamp": 1635939405855, "user_tz": -330, "elapsed": 1981, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d9f90e2c-dfc7-4f0c-83d5-e589e34bfbec"
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/train.parquet.snappy
```

```python id="udqW3JRHldBG" executionInfo={"status": "ok", "timestamp": 1635941161141, "user_tz": -330, "elapsed": 583, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
```

```python colab={"base_uri": "https://localhost:8080/", "height": 439} id="buhAcg9ylkNI" executionInfo={"status": "ok", "timestamp": 1635939448060, "user_tz": -330, "elapsed": 2471, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="178e9baa-b4bb-4fbf-8a25-4a787fa4dc3d"
trainset = pd.read_parquet('train.parquet.snappy')
trainset
```

```python colab={"base_uri": "https://localhost:8080/", "height": 439} id="8koSYqV6luec" executionInfo={"status": "ok", "timestamp": 1635939504411, "user_tz": -330, "elapsed": 480, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="96d32095-2305-4bbf-f579-362c22d8bb1c"
# There are a total of 260087 rows, we randomly choose 10k data for EDA purpose
trainset_10k = trainset.sample(n=10000, random_state=42)
trainset_10k
```

<!-- #region id="XW90eCGroTpB" -->
### Distribution of items bought
<!-- #endregion -->

<!-- #region id="c_Ib8aERl2qU" -->
We first try the easiest: we want to see the distribution of ppl who bought 0, 1, 2, ... 9 items
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="X_oVYeIEmwAo" executionInfo={"status": "ok", "timestamp": 1635939726780, "user_tz": -330, "elapsed": 1304, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dc31caef-8ff5-4ab1-ecb6-c6ff9f133226"
cnt_dict = {}  # how many ppl bought 0, 1, 2, ... 9 items
for index, row in trainset_10k.iterrows():
    item_ids = row['exposed_items'].split(',')
    is_bought = row['labels'].split(',')
    bought_amount = 0
    for i in range(0, len(is_bought)):
        if is_bought[i] == '1':
            bought_amount += 1
    if bought_amount not in cnt_dict:
        cnt_dict[bought_amount] = 1
    else:
        cnt_dict[bought_amount] += 1

cnt_dict
```

```python colab={"base_uri": "https://localhost:8080/"} id="8KhdKPfUm7ct" executionInfo={"status": "ok", "timestamp": 1635939800342, "user_tz": -330, "elapsed": 525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c3ffa312-6dc3-4059-aab2-a1106211bda7"
data_display = []
for i in range(0, 10):
    data_display.append(cnt_dict[i])

data_display
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="U-u-cgkGmTYY" executionInfo={"status": "ok", "timestamp": 1635939805569, "user_tz": -330, "elapsed": 628, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1c5b1391-2186-42be-e113-5d467817b94f"
plt.plot(data_display, color = 'r')
plt.title('Distribution of People who bought X items')
plt.xlabel('# of Items')
plt.ylabel('# of People')
plt.grid(True)
plt.show()
```

<!-- #region id="AqZgla3DmzmS" -->
So most people tend to buy 8 items, 2671 ppl out of 10k.
<!-- #endregion -->

<!-- #region id="JUBMbQvroPhr" -->
### Distribution of user protrait

choose the top 10 user protraits as classifying features
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 219, "referenced_widgets": ["be7df1fa92244c30b59244c3e25a0834", "7a18956fc0254c998ae7a338e6b3fb56", "a0257a7eafdb447c8e57a8b2b7fd7cac", "a99af6eaf6ca4d4b8458285c30253c1a", "1fe34b85f3e44783bffdf835d8e6e665", "a09ae6b986ee461d8aba100886528c63", "17ddf05957704f94b1981942d432e6cc", "0bd9d0bb30954faea82bd562a682a976", "1cd211f33f9a42a8993144dcda140e6e", "e6f829f7ab7847c2b5036c93339f8213", "084eb82f4d1d4331875143061b1b6005"]} id="tPVCtuJUn5Ro" executionInfo={"status": "ok", "timestamp": 1635941182809, "user_tz": -330, "elapsed": 1307, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b2d1c1cf-57e1-4577-f444-5a8265562a8e"
cnt_dict = {}  
for index, row in tqdm(trainset_10k.iterrows(), total=len(trainset_10k)):
    user_protraits = row['user_protrait'].split(',')
    for user_protrait in user_protraits:
        if user_protrait not in cnt_dict:
            cnt_dict[user_protrait] = 1
        else:
            cnt_dict[user_protrait] += 1

user_protrait_cnt_dict = cnt_dict
user_protrait_cnt_list = [(k,v) for k, v in sorted(user_protrait_cnt_dict.items(), key=lambda item: item[1], reverse=True)]

top_10_features = user_protrait_cnt_list[:10]
top_10_features
```

```python colab={"base_uri": "https://localhost:8080/", "height": 315} id="PBuHF8t9ohxm" executionInfo={"status": "ok", "timestamp": 1635940189512, "user_tz": -330, "elapsed": 562, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6be93b48-37a5-4409-ece4-cf5c65f3a5b2"
fig,ax = plt.subplots()
ax.set_xticks([i for i in range(0, 10)])
ax.set_xticklabels([x[0] for x in top_10_features])

plt.plot([x[1] for x in top_10_features], color = 'r')
plt.xticks(rotation=45)
plt.grid(True)
plt.title('Distribution of User Protrait')
plt.xlabel('user protraits')
plt.ylabel('People with this protrait')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["d72fbcdb0e1e4c01bed4a6e1dda155dc", "24e48b0caa8841429081375cf4b2efd9", "48d6e268a5fe45e5bedf0b33f3d0bbee", "d285e4ec27e440f48fbd453d413e8c0a", "60d22f5b5e6242199c110dc472b2effd", "6ffdcad9ecd54976a2cd956f79103f6f", "c38b8fb541724c1687446da0c617847f", "5a3ad650616d4cf7ad24e0b5e56a67b1", "1b63c83eacb847c0aef62d1fefe76308", "bc86bcf1cf3244cbaa74835b6e1d4d2c", "808d9160060c4d4cb01e94bbcad18996"]} id="5tbmspOzojv8" executionInfo={"status": "ok", "timestamp": 1635942341682, "user_tz": -330, "elapsed": 1128496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ab2d6181-969d-4394-ce56-5a50b6e8102f"
columns = ['user_id'] + [x[0] for x in top_10_features] + ['item_' + str(i) for i in range(1, 382)]
processed_trainset = pd.DataFrame([], columns = columns)
cnt = 3

for index, row in tqdm(trainset_10k.iterrows(), total=len(trainset_10k)):
    user_protraits = row['user_protrait'].split(',')
    user_id = str(row['user_id'])
    to_insert = {}

    # mark features
    for column in processed_trainset:
        if column == 'user_id':
            to_insert['user_id']=user_id
        elif not column.startswith('item_'):
            if column in user_protraits:
                to_insert[column]=1
            else:
                to_insert[column]=0
                
    # mark items
    exposed_items = row['exposed_items'].split(',')
    labels = row['labels'].split(',')

    for i in range(0, len(exposed_items)):
        cur_item = 'item_' + exposed_items[i]
#         print(cur_item)
#         print(labels[i])
        to_insert[cur_item] = labels[i]  
    
#     for i in range(1, 382):
#         cur_item_id = 'item_' + str(i)
#         if cur_item_id not in to_insert:
#             to_insert[cur_item_id] = np.nan
    # print(to_insert)
    processed_trainset = processed_trainset.append(to_insert, ignore_index=True)
#     cnt -= 1
    if cnt <= 0:
        break
```

```python colab={"base_uri": "https://localhost:8080/", "height": 439} id="pyPVceVmroBt" executionInfo={"status": "ok", "timestamp": 1635942346750, "user_tz": -330, "elapsed": 473, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9ba49173-6b36-4a17-e3cc-389d333eed3e"
processed_trainset
```

```python id="Zb1fZi8mX6jw" executionInfo={"status": "ok", "timestamp": 1635952623618, "user_tz": -330, "elapsed": 534, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
processed_trainset.to_parquet('processed_trainset.parquet.snappy', compression='snappy')
```

```python id="4oYq550lrpWP" executionInfo={"status": "ok", "timestamp": 1635942358558, "user_tz": -330, "elapsed": 1292, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from sklearn.ensemble import RandomForestClassifier

cur_item = 'item_1'

cur_Xy = processed_trainset[~processed_trainset[cur_item].isnull()]
train_test_perc = 0.8
cur_len = len(cur_Xy)

cur_Xy_train = cur_Xy.head(int(cur_len * train_test_perc))
cur_Xy_test = cur_Xy.tail(cur_len - int(cur_len * train_test_perc))
```

```python colab={"base_uri": "https://localhost:8080/"} id="zF_a3l_Wt-vy" executionInfo={"status": "ok", "timestamp": 1635942360539, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="993f9397-8390-4e4c-fc6e-d36dd07d0b77"
print(cur_len)
print(len(cur_Xy_train))
print(len(cur_Xy_test))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 439} id="J3tAs0cduAO7" executionInfo={"status": "ok", "timestamp": 1635942362573, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="041d3dda-0cda-409d-e2c0-036dabf94a86"
cur_Xy_train
```

```python id="Q2MRsTeeuB2b" executionInfo={"status": "ok", "timestamp": 1635942364879, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
top_10_feature_ids = [x[0] for x in top_10_features]
X_train = cur_Xy_train[[x[0] for x in top_10_features]]
y_train = cur_Xy_train[cur_item]

X_test = cur_Xy_test[[x[0] for x in top_10_features]]
y_test = cur_Xy_test[cur_item]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="R10b3P9vuCgk" executionInfo={"status": "ok", "timestamp": 1635942373629, "user_tz": -330, "elapsed": 938, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="72aa67bc-937c-42e6-c48f-a9e8c1434b66"
X_train
```

```python colab={"base_uri": "https://localhost:8080/"} id="5gfVXTuguDqo" executionInfo={"status": "ok", "timestamp": 1635942373631, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c491f532-d986-4210-e29c-a6f1b4080b1f"
y_train
```

```python colab={"base_uri": "https://localhost:8080/"} id="XyOqRd3JuHZI" executionInfo={"status": "ok", "timestamp": 1635942386516, "user_tz": -330, "elapsed": 605, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a90192a8-256c-4ed0-f8ce-653bf91d1a55"
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

prediction_accuracy = (predictions==y_test).sum() / len(y_test)
prediction_accuracy
```

```python id="IHDNw5s5uKAy" executionInfo={"status": "ok", "timestamp": 1635942404553, "user_tz": -330, "elapsed": 11381, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from sklearn.ensemble import RandomForestClassifier

print_list = []

all_rf = {}
accuracy_dict = {}
cnt = 10
for i in range(1, 382):
    cur_item = 'item_' + str(i)
    cur_Xy = processed_trainset[~processed_trainset[cur_item].isnull()]
    if len(cur_Xy) == 0:
        print_list.append('Item {} never showed up.'.format(i))
        continue
    if len(cur_Xy) < 10:
        print_list.append('Item {} showed up less than 10 times.'.format(i))
        continue    
    train_test_perc = 0.8
    cur_len = len(cur_Xy)

    cur_Xy_train = cur_Xy.head(int(cur_len * train_test_perc))
    cur_Xy_test = cur_Xy.tail(cur_len - int(cur_len * train_test_perc))    
    X_train = cur_Xy_train[[x[0] for x in top_10_features]]
    y_train = cur_Xy_train[cur_item]

    X_test = cur_Xy_test[[x[0] for x in top_10_features]]
    y_test = cur_Xy_test[cur_item]
    
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)    
    
    accuracy = (predictions==y_test).sum() / len(y_test)
    accuracy_dict[i] = accuracy
    print_list.append('Item {} has accuracy {}'.format(i, accuracy))
    
    X = cur_Xy[[x[0] for x in top_10_features]]
    y = cur_Xy[cur_item]
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    
    all_rf[i] = model
        
#     cnt -= 1
    if cnt <=0:
        break
```

```python colab={"base_uri": "https://localhost:8080/"} id="hTuIwEc2ukdr" executionInfo={"status": "ok", "timestamp": 1635942404555, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="052df637-8c7f-4de0-a086-6a890c223ad9"
print(', '.join(print_list))
```

```python colab={"base_uri": "https://localhost:8080/"} id="JbBp0T-euTvd" executionInfo={"status": "ok", "timestamp": 1635942419803, "user_tz": -330, "elapsed": 475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c2817053-7195-458b-8e89-a13c1f2d05bb"
len(accuracy_dict.keys())
```

```python colab={"base_uri": "https://localhost:8080/"} id="PdnC1WEquVG-" executionInfo={"status": "ok", "timestamp": 1635942580189, "user_tz": -330, "elapsed": 648, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="566dcc56-50c2-4e68-8470-e2e6a21f1736"
print(', '.join([str(k)+':'+str(v.round(2)) for k,v in accuracy_dict.items()]))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 456} id="uqAw5sDFxE_J" executionInfo={"status": "ok", "timestamp": 1635942680066, "user_tz": -330, "elapsed": 1901, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6b44246d-44fe-4891-dff5-26195142ddc9"
!wget -q --show-progress https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/track1_testset.parquet.snappy
testset = pd.read_parquet('track1_testset.parquet.snappy')
testset
```

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["82deb732d45c4a6b9de1944c4e318e65", "d23e0306035f40f68050416ecf30658f", "f9c6b16163624b2d9d9da61d69579378", "22bd3675110f4646906088c3d82fbb83", "2ba15ed1cda94c8199e9708255c6d237", "0ae4bce22a6241fab4ad946f35a7206a", "e42ea14ff1754094ada55e17bed63bbe", "1e26163554654f62bfdf0620121bf155", "a4f8c13584234c9aa9e9878b01b87ddd", "571d806250bb458ba587e1aa897df0ef", "b297312704644fcf84f58871652936ef"]} id="NA6qpeFix_zg" executionInfo={"status": "ok", "timestamp": 1635952344606, "user_tz": -330, "elapsed": 9493382, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="71f16d98-385d-4df6-96b1-ff989a454c20"
import time

test_result = pd.DataFrame([], columns = ['id', 'category'])

start_time = int(time.time())
cnt = 1

for index, row in tqdm(testset.iterrows(), total=len(testset)):
#     print(row)
    item_ids = row['exposed_items'].split(',')
    user_protraits = row['user_protrait'].split(',')
#     user_protraits = row['user_protrait'].split(',')    
    cur_X = pd.DataFrame([], columns = top_10_feature_ids)

    to_insert = {}
    for column in cur_X:
        if column in user_protraits:
            to_insert[column]=1
        else:
            to_insert[column]=0
#     print('to_insert')
#     print(to_insert)
    cur_X = cur_X.append(to_insert, ignore_index=True)
#     print('cur_X')        
#     print(cur_X)
    predictions = []
    for item_id in item_ids:
        item_id = int(item_id)
        if item_id in all_rf:
            cur_pred = all_rf[item_id].predict(cur_X)
#             print('Pred item {} is {}'.format(item_id, cur_pred))
            if int(cur_pred) > 0.5:
                cur_pred = 1
            else:
                cur_pred = 0
        else:
            # buy is more than not-buy
            cur_pred = 1
        predictions.append(cur_pred)
#     print('predictions')
#     print(predictions)

#     test_result_to_insert['id'] = row['user_id']
    predictions = [str(p) for p in predictions]
#     test_result_to_insert['category'] = ' '.join(predictions)   
    
    test_result_to_insert = {'id': row['user_id'], 'category': ' '.join(predictions)}   
    test_result = test_result.append(test_result_to_insert, ignore_index=True)
    cnt += 1

print('finally all done')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="2rKkHIPryVjK" executionInfo={"status": "ok", "timestamp": 1635952344609, "user_tz": -330, "elapsed": 31, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="576dc16a-e772-4ee4-ed23-2f9bdce4e75d"
test_result
```

```python id="JImCytZSXs2h" executionInfo={"status": "ok", "timestamp": 1635952577217, "user_tz": -330, "elapsed": 647, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
test_result.to_parquet('test_result.parquet.snappy', compression='snappy')
```

```python colab={"base_uri": "https://localhost:8080/"} id="HPk4v_CxYFdU" executionInfo={"status": "ok", "timestamp": 1635952682590, "user_tz": -330, "elapsed": 16676, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1978cf77-f691-4180-9614-fefcccf36a5e"
import os
project_name = "ieee21cup-recsys"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', branch)

!cp -r /content/drive/MyDrive/git_credentials/. ~
!mkdir "{project_path}"
%cd "{project_path}"
!git init
!git remote add origin https://github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
!git checkout -b "{branch}"

!mkdir -p data/silver/T604426
!mv /content/test_result.parquet.snappy data/silver/T604426
!mv /content/processed_trainset.parquet.snappy data/silver/T604426

!git add .
!git commit -m 'commit'
!git push origin main
```
