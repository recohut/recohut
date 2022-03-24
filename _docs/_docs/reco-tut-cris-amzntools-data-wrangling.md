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

```python id="Dg8frDmMWhHA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628058103412, "user_tz": -330, "elapsed": 3824, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="433ed277-a689-44c1-a5ad-29c60de6deac"
import os
project_name = "reco-tut-cris"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

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

```python id="0wm-JFXm03Jg" executionInfo={"status": "ok", "timestamp": 1628058125165, "user_tz": -330, "elapsed": 463, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pdb
import os
import sys
import csv
import json
import numpy as np
```

```python colab={"base_uri": "https://localhost:8080/"} id="7rykBE_71hRd" executionInfo={"status": "ok", "timestamp": 1628058343123, "user_tz": -330, "elapsed": 5786, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a45d449e-5d4b-4e64-f113-f4fde39de1f6"
!cd ./data/bronze && sh load.sh
```

```python colab={"base_uri": "https://localhost:8080/"} id="iSt3QID51r-J" executionInfo={"status": "ok", "timestamp": 1628058394025, "user_tz": -330, "elapsed": 629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a2bcc0af-3458-4dd1-a51f-531fa7ba98b4"
filepath = './data/bronze/amazon_tools.json'
!head -5 $filepath
```

```python id="3tunXnqQ1Sui" executionInfo={"status": "ok", "timestamp": 1628058238772, "user_tz": -330, "elapsed": 602, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def split_by_month(data):
    trn, vld, tst = [], [], []

    numdata = len(data)

    data.sort(key=lambda x:x[-1]) # sort data by date (in unix timestamp)
    
    times = np.array([float(i[-1]) for i in data])
    
    oldest_time = max(times)
    start_test_time = oldest_time - 30 * 24 * 60 * 60 # (30 days)
    start_valid_time = start_test_time - 30 * 24 * 60 * 60 # (30 days)
    
    tst_idx = times >= start_test_time
    vld_idx = (times >= start_valid_time) * (times < start_test_time)
    trn_idx = ~(tst_idx + vld_idx)

    trn_end_idx = np.where(trn_idx == True)[0].max()
    vld_end_idx = np.where(vld_idx == True)[0].max()

    trn = data[:trn_end_idx]
    vld = data[trn_end_idx:vld_end_idx]
    tst = data[vld_end_idx:]
    
    # Filter out new (cold-start) users and items
    trnusers = set([i[0] for i in trn])
    trnitems = set([i[1] for i in trn])

    vld = [row for row in vld if (row[0] in trnusers and row[1] in trnitems)]
    tst = [row for row in tst if (row[0] in trnusers and row[1] in trnitems)]

    print('\nTraining data:\t\t {}'.format(len(trn)))
    print('Validation data:\t {}'.format(len(vld)))
    print('Test data:\t\t {}'.format(len(tst)))
    print('\n# of total data:\t {} / {}'.format(len(trn) + len(vld) + len(tst), len(data)))

    return trn, vld, tst
```

```python id="lMLlR9cO1cr_" executionInfo={"status": "ok", "timestamp": 1628058432438, "user_tz": -330, "elapsed": 1974, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
data = [json.loads(l) for l in open(filepath)]
mydata = [[l['reviewerID'], l['asin'], l['overall'], int(float(l['unixReviewTime']))] for l in data]
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ps1QVgX02FOH" executionInfo={"status": "ok", "timestamp": 1628058456586, "user_tz": -330, "elapsed": 520, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0a4d2b8b-47df-447f-aa8e-c09474ba0054"
mydata[0:5]
```

```python colab={"base_uri": "https://localhost:8080/"} id="NeQMM4J52EZC" executionInfo={"status": "ok", "timestamp": 1628058505636, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c03a832f-96fa-4691-a542-da16f3b7f7a3"
trndata, vlddata, tstdata = split_by_month(mydata)
```

```python id="KIT9MJaw4enW" executionInfo={"status": "ok", "timestamp": 1628059078969, "user_tz": -330, "elapsed": 421, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="89dacf66-1a59-4a5a-e90a-8953f9b1d3a1" colab={"base_uri": "https://localhost:8080/"}
trndata[:5]
```

```python id="HRKn5NIE2VAo" executionInfo={"status": "ok", "timestamp": 1628058799227, "user_tz": -330, "elapsed": 518, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
basename = filepath.split('/')[-1].split('.')[0]

dirname = './data/silver/'+basename
if not os.path.exists(dirname): os.makedirs(dirname)
```

```python id="6yNs5Seldl4j" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628058803952, "user_tz": -330, "elapsed": 723, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6da63d4-20cf-4055-ed3f-3d49e877a4d5"
# Save the dataset in csv format
writer = csv.writer(open(dirname+'/train.csv', 'w'))
writer.writerows(trndata)

writer = csv.writer(open(dirname+'/valid.csv', 'w'))
writer.writerows(vlddata)

writer = csv.writer(open(dirname+'/test.csv', 'w'))
writer.writerows(tstdata)

print('\nDone\n')
```

```python colab={"base_uri": "https://localhost:8080/"} id="F2Ow7Lgd3Sx5" executionInfo={"status": "ok", "timestamp": 1628058882597, "user_tz": -330, "elapsed": 633, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ab8c466c-d40b-420d-91fa-625501586e90"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="LydFTeC93wvM" executionInfo={"status": "ok", "timestamp": 1628058936761, "user_tz": -330, "elapsed": 2263, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0e8aed07-1ad7-4a0d-8a99-5601738f40f2"
!git add . && git commit -m 'ADD data in silver layer amazon tools' && git push origin main
```
