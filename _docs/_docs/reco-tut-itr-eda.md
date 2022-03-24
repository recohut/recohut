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

```python id="xWTTsFsu3idp" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628093088840, "user_tz": -330, "elapsed": 2503, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1a2eaa13-e78f-4a5a-8cbc-b352764b360d"
import os
project_name = "reco-tut-itr"; branch = "main"; account = "sparsh-ai"
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

```python id="VcOsQNy36PeN" executionInfo={"status": "ok", "timestamp": 1628094208798, "user_tz": -330, "elapsed": 1189, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

```python colab={"base_uri": "https://localhost:8080/"} id="xwqtqVoK6bdx" executionInfo={"status": "ok", "timestamp": 1628093217613, "user_tz": -330, "elapsed": 660, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="47d79fa4-c2b5-4c47-b9d2-cb0ba30bf30a"
files = glob.glob('./data/bronze/*')
files
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="onf-sANe6t-6" executionInfo={"status": "ok", "timestamp": 1628093570347, "user_tz": -330, "elapsed": 594, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="137eb833-3dfd-41c3-fc4f-a95051f343a9"
df1 = pd.read_parquet(files[0])
df1.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Hl4SId5760J2" executionInfo={"status": "ok", "timestamp": 1628093570766, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9ed13b19-c16f-4944-8b31-b0cf95b8fe84"
df1.info()
```

<!-- #region id="lLtCrzac65QM" -->
> Notes
- There are total 2890 records but rating info for only 1777 is availble. We will keep only not-null rating records.
- userId and itemId need to be categorical.
- timestamp data type correction.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1PgkjYZL7zXV" executionInfo={"status": "ok", "timestamp": 1628093572389, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7984a3d2-6fd8-4d61-9b0c-f200e0b59a2b"
df1 = df1.dropna(subset=['rating'])
df1.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="wBGlaDnU7nyu" executionInfo={"status": "ok", "timestamp": 1628093661376, "user_tz": -330, "elapsed": 507, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="adc30624-eb29-497f-8ce7-d9452123912f"
df1 = df1.astype({'userId': 'str', 'itemId': 'str'})
df1.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="VHAspwh47lp5" executionInfo={"status": "ok", "timestamp": 1628093869037, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7a29a961-f8c6-42c9-e41c-baeae560961f"
df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='s')
df1.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 190} id="Z2ng08Z17j2-" executionInfo={"status": "ok", "timestamp": 1628093900789, "user_tz": -330, "elapsed": 1090, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aaf791dd-70ed-4dd3-87fa-9a1cee32b9b0"
df1.describe(include='all', datetime_is_numeric=True).T
```

<!-- #region id="YavgmS7N8r7T" -->
> Notes
- There are only 10 users providing ratings for 289 visiting places
- Mean rating is 3.5
- Data looks pretty old (year 1997-98)
- Timespan is of 8 months (almost)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 277} id="rNw-yzOz-cio" executionInfo={"status": "ok", "timestamp": 1628094423467, "user_tz": -330, "elapsed": 1093, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c08cf5e4-598a-4ba9-e465-25968a1c2717"
fig, ax = plt.subplots(figsize=(16,4))
df1.groupby(df1['timestamp'])['rating'].count().plot(kind='line', ax=ax)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 271} id="lSNV5NPr_dTx" executionInfo={"status": "ok", "timestamp": 1628094560812, "user_tz": -330, "elapsed": 532, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e3b9858c-f78f-4369-88af-c3bf7b09acd8"
fig, ax = plt.subplots(figsize=(8,4))
df1.rating.value_counts().plot(kind='bar', ax=ax)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="TrmwtD_L_r00" executionInfo={"status": "ok", "timestamp": 1628094583634, "user_tz": -330, "elapsed": 408, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b1fbfc95-5cf8-4906-c108-ef1c8bed7932"
df1.rating.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="9GAOXg5N_7Da" executionInfo={"status": "ok", "timestamp": 1628094648062, "user_tz": -330, "elapsed": 457, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9d93e9c1-aa77-4c77-c9a0-71e0eb38e93e"
df2 = pd.read_parquet(files[1])
df2.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="b9z8ItuOAMkv" executionInfo={"status": "ok", "timestamp": 1628094667390, "user_tz": -330, "elapsed": 1033, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3bb21acc-a259-4635-c219-2f62ff1df18d"
df2.info()
```

<!-- #region id="AGyZ1kojAPHH" -->
> Notes
- We have ratings for 289 items but here only 286 items are available, so we need to investigate and correct this mismatch
- Also correct the itemId data type here also
- No missing values, quite strange but ok
<!-- #endregion -->

```python id="GnmkENOSBM15" executionInfo={"status": "ok", "timestamp": 1628094994924, "user_tz": -330, "elapsed": 432, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df2 = df2.astype({'itemId': 'str'})
```

```python colab={"base_uri": "https://localhost:8080/"} id="1uhtA7bVBhWS" executionInfo={"status": "ok", "timestamp": 1628095137686, "user_tz": -330, "elapsed": 585, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ef1c6a69-323a-419f-8af1-35b40482df6d"
items_df1 = df1.itemId.unique()
items_df2 = df2.itemId.unique()

set(items_df1) - set(items_df2)
```

<!-- #region id="_5ZE-BRCCD8G" -->
> Notes
- Since we do not have metadata for these three items, let's see how many ratings are there for these, if not much, we can remove the records, otherwise, we will remove later if we train any hybrid model that used both rating and item metadata information.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LAuZodu-Cluf" executionInfo={"status": "ok", "timestamp": 1628095368102, "user_tz": -330, "elapsed": 564, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="58dd68c3-d9a0-4727-e8d0-9ed1fdbdcafb"
df1[df1.itemId.isin(list(set(items_df1) - set(items_df2)))].shape
```

<!-- #region id="3w44jUtDCsdH" -->
> Notes
- 19 out of 1777, let's remove it
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Is51m5v2DCx8" executionInfo={"status": "ok", "timestamp": 1628095428447, "user_tz": -330, "elapsed": 474, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b416f88-1d60-4e2f-8778-5891aad6c235"
df1 = df1[~df1.itemId.isin(list(set(items_df1) - set(items_df2)))]
df1.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 638} id="IEF0Pbb2DLGB" executionInfo={"status": "ok", "timestamp": 1628095448673, "user_tz": -330, "elapsed": 470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ffed73c2-d763-408b-d5e6-47c240584619"
df2.describe().T
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="O5Cg9yD1DOem" executionInfo={"status": "ok", "timestamp": 1628095498208, "user_tz": -330, "elapsed": 465, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1dbce9f3-6930-4ca0-ceee-b54cd7e0d859"
df2.describe(include='O').T
```

<!-- #region id="mSDcaLvNEvOv" -->
> Notes
- Seems like creator of this dataset already preprocessed some fields, created one-hot encodings. We will remove these columns, to make things a little less messy and will do this type of encoding during modeling data preparation.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1ClNH5y0DcN6" executionInfo={"status": "ok", "timestamp": 1628095857082, "user_tz": -330, "elapsed": 647, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6e17f056-61f6-428e-b2ef-0d0d8c34869c"
df2 = df2.loc[:, ~df2.columns.str.startswith('travel_')]
df2 = df2.loc[:, ~df2.columns.str.startswith('religion_')]
df2 = df2.loc[:, ~df2.columns.str.startswith('season_')]
df2.info()
```

```python id="UDnRIBvBE4e_" executionInfo={"status": "ok", "timestamp": 1628095970514, "user_tz": -330, "elapsed": 646, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir ./data/silver
df1.to_parquet('./data/silver/rating.parquet.gz', compression='gzip')
df2.to_parquet('./data/silver/items.parquet.gz', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/"} id="2cEQB9teFS43" executionInfo={"status": "ok", "timestamp": 1628096002332, "user_tz": -330, "elapsed": 414, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f24f2f85-dfb1-4beb-d32c-347648475d09"
!git status
```

```python id="VQ87mjICFXTP" executionInfo={"status": "ok", "timestamp": 1628096023844, "user_tz": -330, "elapsed": 676, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f54ebfbc-f5d0-4615-d701-81e9fa9eb78d" colab={"base_uri": "https://localhost:8080/"}
!git add . && git commit -m 'commit' && git push origin main
```
