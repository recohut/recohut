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

```python id="fbhn_BlE4KbY" executionInfo={"status": "ok", "timestamp": 1629641034562, "user_tz": -330, "elapsed": 597, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-rrt"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="H9fA0_x76d7z" executionInfo={"status": "ok", "timestamp": 1629641038082, "user_tz": -330, "elapsed": 3079, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6723c0b5-54d8-48d1-cf12-94b1ea2dd3b3"
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

```python id="S4knsUJY6d76" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629643850839, "user_tz": -330, "elapsed": 723, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4d9b29ce-6bf7-4f8b-e161-ba25a447171d"
!git status
```

```python id="EEiOf79m6d77" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629643873348, "user_tz": -330, "elapsed": 1341, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9dba2253-1cf4-4fe0-9e78-c71f50dd0b59"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="pyNXPQVhLM64" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vdg76ndrL-7W" executionInfo={"status": "ok", "timestamp": 1629641362528, "user_tz": -330, "elapsed": 113229, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b30dded-f541-4713-ccb5-f3eecff3692c"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d yelp-dataset/yelp-dataset
```

```python colab={"base_uri": "https://localhost:8080/"} id="NXX1NrrdMNjg" executionInfo={"status": "ok", "timestamp": 1629641645056, "user_tz": -330, "elapsed": 276281, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="20d61ccb-58c4-48de-cacd-66ad024d08d6"
!mkdir /content/temp && unzip yelp-dataset.zip -d /content/temp
```

```python id="XqdH6whAMFdp" executionInfo={"status": "ok", "timestamp": 1629641646552, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
from tqdm.notebook import tqdm
import json
```

```python id="yALFent8OUfb"
# line_count = len(open("/content/temp/yelp_academic_dataset_review.json").readlines())
line_count = 8635403
user_ids, business_ids, stars, dates = [], [], [], []
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["6145f0f958284a53939b5a22a33568e9", "df953f8eba624ec68acc51e2aabe3f28", "19c3b3efd6b94659a2a5ad7e731a2128", "0a59e4887d3d47168e28850d49233e01", "0b886d254ad64f0eab16354c708be46a", "dab18fee213b4cf49fd1bdb67a9f99c3", "16f7733b9be540a8a261f6cf433501d7", "c9f9cd39cc9b43eb82a43eef569f4ad2", "11ffa93abf314abcbced942a67b3fb22", "ef601b781b114b8eabdf385c863764fd", "def5947fd9914420a317ce61f7658ef0"]} id="bZSnZiA3MICq" executionInfo={"status": "ok", "timestamp": 1629642103010, "user_tz": -330, "elapsed": 200720, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="da483956-82e1-404d-def6-17f2d06bd6a7"
with open("/content/temp/yelp_academic_dataset_review.json") as f:
    for line in tqdm(f, total=line_count):
        blob = json.loads(line)
        user_ids += [blob["user_id"]]
        business_ids += [blob["business_id"]]
        stars += [blob["stars"]]
        dates += [blob["date"]]
ratings_ = pd.DataFrame(
    {"user_id": user_ids, "business_id": business_ids, "rating": stars, "date": dates}
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="oIl1tSyxNMcu" executionInfo={"status": "ok", "timestamp": 1629642176841, "user_tz": -330, "elapsed": 778, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc03d25f-c1f8-41dd-c4d4-63a0e3bd09fc"
ratings_.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="C9luRF21Pitb" executionInfo={"status": "ok", "timestamp": 1629642185184, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0d468ee1-887e-485a-86dd-c65524ee84fb"
ratings_.info()
```

```python id="sRyiDWx_PlPe" executionInfo={"status": "ok", "timestamp": 1629642281144, "user_tz": -330, "elapsed": 8001, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir -p ./data/bronze/yelp
ratings_.to_parquet('./data/bronze/yelp/interactions.parquet.gzip', compression='gzip')
```

```python id="3JLYwV5UP65q" executionInfo={"status": "ok", "timestamp": 1629642590677, "user_tz": -330, "elapsed": 30178, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from sklearn import preprocessing
le_user = preprocessing.LabelEncoder()
ratings_['user_id'] = le_user.fit_transform(ratings_['user_id'])
le_item = preprocessing.LabelEncoder()
ratings_['business_id'] = le_item.fit_transform(ratings_['business_id'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="yy1Pj922QF5l" executionInfo={"status": "ok", "timestamp": 1629642602033, "user_tz": -330, "elapsed": 429, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="58623c3a-3ceb-467d-916c-892f8422931e"
ratings_.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="fn_T60OxRLHf" executionInfo={"status": "ok", "timestamp": 1629642632058, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e3d34689-7e01-4157-e632-8f5c391f775b"
ratings_.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="CursLEHcRSOq" executionInfo={"status": "ok", "timestamp": 1629642751221, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="49aad433-5357-4789-d433-9d2975632875"
ratings_['date'] = pd.to_datetime(ratings_['date'])
ratings_.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="9NvEIwMnRoCW" executionInfo={"status": "ok", "timestamp": 1629642753676, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cedbe691-f5c2-414f-ea4c-6ced0f6d4fb3"
ratings_.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="wRqlA7g-RshQ" executionInfo={"status": "ok", "timestamp": 1629642796697, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d90979fd-8d17-4d6f-e665-f219de09f6d5"
ratings_.rating.value_counts()
```

```python id="OPOPqhugR6Ye" executionInfo={"status": "ok", "timestamp": 1629642837030, "user_tz": -330, "elapsed": 616, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
ratings_['rating'] = ratings_['rating'].astype('uint8')
```

```python colab={"base_uri": "https://localhost:8080/"} id="YjzWExFPSCuk" executionInfo={"status": "ok", "timestamp": 1629642844879, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f2c70dc6-775d-492f-de4d-31a3b1a6e402"
ratings_.info()
```

```python id="W-t7cbINSGcX" executionInfo={"status": "ok", "timestamp": 1629642910472, "user_tz": -330, "elapsed": 24782, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
ratings_.to_parquet('./data/bronze/yelp/interactions2.parquet.gzip', compression='gzip')
ratings_.to_parquet('./data/bronze/yelp/interactions2.parquet.snappy', compression='snappy')
```

<!-- #region id="HlOzyVZGTF69" -->
> Note: Rename it back to interactions and keep only snappy format. Let's export the label encoders also.
<!-- #endregion -->

```python id="kereaJpnTpt6" executionInfo={"status": "ok", "timestamp": 1629643573430, "user_tz": -330, "elapsed": 2169, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pickle
pickle.dump(le_user, open('./data/bronze/yelp/le_user.pkl', 'wb'))
pickle.dump(le_item, open('./data/bronze/yelp/le_item.pkl', 'wb'))
```

<!-- #region id="M0T9kK2vUYFT" -->
Start DVC
<!-- #endregion -->

```python id="CtBKt3PoVBMF"
!pip install -q git+https://github.com/sparsh-ai/recochef
!pip install -U -q PyDrive dvc dvc[gdrive]
from recochef.utils.gdrive import *
drive_handler = GoogleDriveHandler()
test_subfolder_id = drive_handler.create_folder('reco-tut-rrt')
```

```python colab={"base_uri": "https://localhost:8080/"} id="vrLj-4NAVCYH" executionInfo={"status": "ok", "timestamp": 1629643832706, "user_tz": -330, "elapsed": 42312, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7c319298-95f0-4416-9700-6a8c5f289286"
!dvc init
!dvc remote add -d myremote gdrive://"{test_subfolder_id}"
!dvc add ./data/bronze/yelp/interactions.parquet.snappy
!dvc add ./data/bronze/yelp/le_user.pkl
!dvc add ./data/bronze/yelp/le_item.pkl
!dvc commit
!dvc push
```

```python id="OPhCP7psWL-q"
!apt-get install tree
```

```python id="LfvQ9yvRWQ_5" executionInfo={"status": "ok", "timestamp": 1629643950889, "user_tz": -330, "elapsed": 744, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aa0b01c6-d5e2-4d73-9443-b30645159ceb" colab={"base_uri": "https://localhost:8080/"}
!tree .
```
