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

```python id="JBGrCqRDgTzg" executionInfo={"status": "ok", "timestamp": 1629260728733, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-ypr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="adbscBjegAvW" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629260731599, "user_tz": -330, "elapsed": 2187, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3413053a-ccbf-4499-b915-afccfabc83b7"
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

```python id="etrmX9tqgAvZ"
!git status
```

```python id="HCB-Yq-ygAva"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="cArSWM9Jge3y" -->
---
<!-- #endregion -->

<!-- #region id="23MoJMItrX7U" -->
### Data loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pfl4MvWIgczD" executionInfo={"status": "ok", "timestamp": 1629260836599, "user_tz": -330, "elapsed": 45691, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d35a3278-40c6-4583-cc16-0343788dad44"
!curl https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z --output data.7z
!7z e data.7z -o/content/rsc15
!rm data.7z
```

```python id="Swy-6EFhhGCu" executionInfo={"status": "ok", "timestamp": 1629262723917, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

```python id="sNZANLTwhm_V" executionInfo={"status": "ok", "timestamp": 1629262755876, "user_tz": -330, "elapsed": 792, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
src_path = '/content/rsc15/'
!mkdir -p /content/data
dst_path   = '/content/data/'
```

```python id="uEwiPVp7on1G"
!git clone https://github.com/pcerdam/KerasGRU4Rec --single-branch src
```

<!-- #region id="4yXCK7WMrbIZ" -->
### GRU4Rec model
<!-- #endregion -->

```python id="uMXH_hXAhM5h" executionInfo={"status": "ok", "timestamp": 1629262853377, "user_tz": -330, "elapsed": 648, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def process(src_path, dst_path):
  data = pd.read_csv(src_path + 'yoochoose-clicks.dat', sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
  data.columns = ['SessionId', 'TimeStr', 'ItemId']

  # Add timestamp
  data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
  del(data['TimeStr'])

  # Filter by session size > 1
  session_lengths = data.groupby('SessionId').size()  
  data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
  
  # Filter by "known" (appears >= 5) items
  item_supports = data.groupby('ItemId').size()
  data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]
  
  # Filter by session size > 2
  session_lengths = data.groupby('SessionId').size()
  data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]

  # Test split is entire last day
  tmax = data.Time.max()
  session_max_times = data.groupby('SessionId').Time.max()
  
  session_train = session_max_times[session_max_times < tmax-86400].index
  train = data[np.in1d(data.SessionId, session_train)]
  
  session_test = session_max_times[session_max_times >= tmax-86400].index
  test = data[np.in1d(data.SessionId, session_test)]
  test = test[np.in1d(test.ItemId, train.ItemId)]
  tslength = test.groupby('SessionId').size()
  test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
  print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
  test.to_csv(dst_path + 'test.tsv', sep='\t', index=False)

  # Validation: second to last day. Train: remainder
  tmax = train.Time.max()
  session_max_times = train.groupby('SessionId').Time.max()
  session_train = session_max_times[session_max_times < tmax-86400].index
  session_valid = session_max_times[session_max_times >= tmax-86400].index

  train_tr = train[np.in1d(train.SessionId, session_train)]
  valid = train[np.in1d(train.SessionId, session_valid)]
  valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
  tslength = valid.groupby('SessionId').size()
  valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
  print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
  valid.to_csv(dst_path + 'validation.tsv', sep='\t', index=False)
  
  print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
  train_tr.to_csv(dst_path + 'train.tsv', sep='\t', index=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="HAVgvOe0oiqo" executionInfo={"status": "ok", "timestamp": 1629263577457, "user_tz": -330, "elapsed": 628155, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2c573ecd-61a5-4b67-feae-18d0d2937fd2"
process(src_path, dst_path)  # Takes ~10 minutes
```

```python id="QMvmSr4kon75" executionInfo={"status": "ok", "timestamp": 1629263577459, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def train_fraction(df, fraction, prefix="train", path="./data/"):
    
    length = len(df['ItemId'])
    first_session = df.iloc[length - length//fraction].SessionId
    df = df.loc[df['SessionId'] >= first_session]
    itemids = df['ItemId'].unique()
    n_items = len(itemids)

    print('Fractioned data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(df), df.SessionId.nunique(), df.ItemId.nunique()))
    filename = path + '{}_{}.tsv'.format(prefix, fraction)
    df.to_csv(filename, sep='\t', index=False)
    print("Saved as {}".format(filename))
```

```python colab={"base_uri": "https://localhost:8080/"} id="z2WYKzpYon4W" executionInfo={"status": "ok", "timestamp": 1629263590486, "user_tz": -330, "elapsed": 10371, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="afb92d5f-f653-4c80-88bf-9f78b03c1468"
# Obtain most recent 1/nth fraction of train for demo purposes
fraction = 64
train = pd.read_csv(dst_path + 'train.tsv', sep='\t', dtype={'ItemId':np.int64})
train_fraction(train, fraction)
```

```python colab={"base_uri": "https://localhost:8080/"} id="osRJtNj0qKiS" executionInfo={"status": "ok", "timestamp": 1629263603690, "user_tz": -330, "elapsed": 11969, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1c3dca08-798e-4de2-c2df-650737763c43"
!python src/model/gru4rec.py --epochs 200 --train-path ./data/train_64.tsv --dev-path ./data/validation.tsv --test-path ./data/test.tsv
```

<!-- #region id="oPmtTjJmqUoJ" -->
### Taking advantage of implicit information: Dwell Time

- We can use the time that each item was visited by each user to make better recommendations.

- The assumption is: the longer the dwell time on an item, the more interested the user is in that item.

- Item boosting will be done in the training set.
<!-- #endregion -->

```python id="2_LyiB0OqfAG" executionInfo={"status": "ok", "timestamp": 1629263603708, "user_tz": -330, "elapsed": 46, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def preprocess_df(df):    
    n_items = len(df['ItemId'].unique())
    aux = list(df['ItemId'].unique())
    itemids = np.array(aux)
    itemidmap = pd.Series(data=np.arange(n_items), index=itemids)  # (id_item => (0, n_items))
    
    item_key = 'ItemId'
    session_key = 'SessionId'
    time_key = 'Time'
    
    data = pd.merge(df, pd.DataFrame({item_key:itemids, 'ItemIdx':itemidmap[itemids].values}), on=item_key, how='inner')
    data.sort_values([session_key, time_key], inplace=True)

    length = len(data['ItemId'])
        
    return data
```

```python id="-SNLFLPQqqYb" executionInfo={"status": "ok", "timestamp": 1629263603710, "user_tz": -330, "elapsed": 45, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def compute_dwell_time(df):
    times_t = np.roll(df['Time'], -1) # Take time row
    times_dt  = df['Time']            # Copy, then displace by one
    diffs = np.subtract(times_t, times_dt) # Take pairwise difference
    length = len(df['ItemId'])
    
    # cummulative offset start for each session
    offset_sessions = np.zeros(df['SessionId'].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = df.groupby('SessionId').size().cumsum() 
    offset_sessions = offset_sessions - 1
    offset_sessions = np.roll(offset_sessions, -1)
    
    np.put(diffs.values, offset_sessions, np.zeros((offset_sessions.shape)), mode='raise')

    return diffs
```

```python id="AUZuI4uCquVy" executionInfo={"status": "ok", "timestamp": 1629263603714, "user_tz": -330, "elapsed": 48, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_distribution(dts):
    filtered = np.array(list(filter(lambda x: int(x) != 0, dts)))
    pd_dts = pd.DataFrame(filtered)
    pd_dts.boxplot(vert=False, showfliers=False)
    plt.xlabel("Seconds")
    plt.yticks([])
    plt.title("Dwell time distribution for RSC15 dataset")
    plt.show()
    pd_dts.describe()
```

```python id="IETqc-0WquSL" executionInfo={"status": "ok", "timestamp": 1629263603716, "user_tz": -330, "elapsed": 49, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def join_dwell_reps(df, dt, threshold=2000):
    # Calculate d_ti/threshold + 1 then add column to dataFrame
    dt //= threshold
    dt += 1   
    df['DwellReps'] = pd.Series(dt.astype(np.int64), index=dt.index)
```

```python id="8MLUzq_-quOw" executionInfo={"status": "ok", "timestamp": 1629263603719, "user_tz": -330, "elapsed": 50, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def augment(df):    
    col_names = list(df.columns.values)[:3]
    augmented = np.repeat(df.values, df['DwellReps'], axis=0) 
    augmented = pd.DataFrame(data=augmented[:,:3],
                             columns=col_names)
    dtype = {'SessionId': np.int64, 
             'ItemId': np.int64, 
             'Time': np.float32}
    
    for k, v in dtype.items():
        augmented[k] = augmented[k].astype(v)                  
    
    return augmented
```

```python id="aar6Tn-Fqy5e"
new_df = preprocess_df(train)
dts = compute_dwell_time(new_df)
```

```python id="D9I5TUO1q0M5"
# Visualize
get_distribution(dts)
```

```python id="jka0AFe7q15x"
# threshold is a hyperparameter
join_dwell_reps(new_df, dts, threshold=75)
```

```python id="0OAQKMBMq13K"
# augment the sessions copying each entry an additional (dwellReps[i]-1) times
df_aug = augment(new_df)
df_aug.to_csv("./data/augmented_train.csv", index=False, sep='\t')
```

```python id="M0YqJ07Kq510"
# retrieve 1/n most recent fraction for demo purposes 
# (note it is a smaller fraction due to the inflation process)
fraction = 64*7
train_fraction(df_aug, fraction, prefix='aug_train')
```

```python id="EGfZS1Ihq9rc"
!python src/model/gru4rec.py --epochs 200 --train-path ./data/aug_train_448.tsv --dev-path ./data/validation.tsv --test-path ./data/test.tsv
```
