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

<!-- #region id="4h7ZIJolugC6" -->
### Setup git
<!-- #endregion -->

```python id="-UOOzCs9ukul" executionInfo={"status": "ok", "timestamp": 1627720992593, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
project_name = "reco-tut-arr"; branch = "main"; account = "sparsh-ai"
```

```python colab={"base_uri": "https://localhost:8080/"} id="PYvHGli8ukum" executionInfo={"status": "ok", "timestamp": 1627720997530, "user_tz": -330, "elapsed": 1913, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5c0553af-0d0c-4ae8-cc4c-28e148808f5a"
!cp /content/drive/MyDrive/mykeys.py /content
import mykeys
!rm /content/mykeys.py
path = "/content/" + project_name; 
!mkdir "{path}"
%cd "{path}"
import sys; sys.path.append(path)
!git config --global user.email "arr@recohut.com"
!git config --global user.name  "reco-tut-arr"
!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
```

<!-- #region id="vMxhh-XUu1Ik" -->
## Extract data from Kaggle
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5ArqEml3vPwa" executionInfo={"status": "ok", "timestamp": 1627721109952, "user_tz": -330, "elapsed": 483, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="393deffc-885c-4f8a-860a-df13c3d24b6a"
%cd /content
```

```python colab={"base_uri": "https://localhost:8080/"} id="o6RLgFTsumNS" executionInfo={"status": "ok", "timestamp": 1627721090185, "user_tz": -330, "elapsed": 8186, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5b36a251-dc88-4f73-babf-36d9694b458a"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d calebemelike/akeed-restaurant
```

```python colab={"base_uri": "https://localhost:8080/"} id="FcTJzRTIvKui" executionInfo={"status": "ok", "timestamp": 1627721133316, "user_tz": -330, "elapsed": 1237, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5aa0d24f-49c2-4430-9c68-55136a7dde02"
!unzip akeed-restaurant.zip
```

```python id="Ia86UZi6vYJr" executionInfo={"status": "ok", "timestamp": 1627721219342, "user_tz": -330, "elapsed": 459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="Q6l2jsejvYHS" executionInfo={"status": "ok", "timestamp": 1627721181105, "user_tz": -330, "elapsed": 512, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="57f58fb5-488d-4d1a-fcff-d765ba3f15ef"
train_customers = pd.read_csv('train_customers.csv')
train_customers
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="98N64GFDwRlZ" executionInfo={"status": "ok", "timestamp": 1627721391991, "user_tz": -330, "elapsed": 739, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="24041c9c-d744-4ead-ba1f-efd0bdf78586"
test_customers = pd.read_csv('test_customers.csv')
test_customers
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="3eCbiucFwZLO" executionInfo={"status": "ok", "timestamp": 1627721418296, "user_tz": -330, "elapsed": 628, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="823f4a90-c4e2-4308-aa3d-9c8fcbc4513b"
train_locations = pd.read_csv('train_locations.csv')
train_locations
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="QEyiUcjgwdFK" executionInfo={"status": "ok", "timestamp": 1627721438546, "user_tz": -330, "elapsed": 489, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b5e11d2f-4eeb-4d6b-d994-0c4713b52366"
test_locations = pd.read_csv('test_locations.csv')
test_locations
```

```python colab={"base_uri": "https://localhost:8080/", "height": 677} id="BGfYJBa-wmde" executionInfo={"status": "ok", "timestamp": 1627721469535, "user_tz": -330, "elapsed": 1281, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="77cb6d7a-df44-47e8-c1bc-a07d0b37f198"
orders = pd.read_csv('orders.csv')
orders
```

<!-- #region id="ki4ks4q6wyrh" -->
Some corrections required in this data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="14SpLZPywwML" executionInfo={"status": "ok", "timestamp": 1627721506666, "user_tz": -330, "elapsed": 476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e16e5c14-16df-46c7-b36c-13bc765190dd"
orders.info()
```

```python id="n_-XLOb6w7U2" executionInfo={"status": "ok", "timestamp": 1627721581622, "user_tz": -330, "elapsed": 413, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
orders.iloc[:,[15,16,18,19,20]] = orders.iloc[:,[15,16,18,19,20]].astype('str')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 609} id="SdnyWdNZwl9J" executionInfo={"status": "ok", "timestamp": 1627721585318, "user_tz": -330, "elapsed": 526, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c0742b0b-9bb5-4cfe-c588-ed074593a6e8"
vendors = pd.read_csv('vendors.csv')
vendors
```

<!-- #region id="KR4qCZLSwMTA" -->
## Store as bronze data layer
<!-- #endregion -->

```python id="bz61BwWyvYEu" executionInfo={"status": "ok", "timestamp": 1627722188960, "user_tz": -330, "elapsed": 3176, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
data_path_bronze = "/content/reco-tut-arr/data/bronze/"

if not os.path.exists(data_path_bronze):
    os.makedirs(data_path_bronze)

def store_df(df,name):
    df.to_parquet(os.path.join(data_path_bronze,name+'.parquet.gz'), compression='gzip')

store_df(train_customers, 'train_customers')
store_df(test_customers, 'test_customers')
store_df(train_locations, 'train_locations')
store_df(test_locations, 'test_locations')
store_df(orders, 'orders')
store_df(vendors, 'vendors')
```

<!-- #region id="UwptvQC5wJX6" -->
## Save and commit
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="R5fq2zAgzjL6" executionInfo={"status": "ok", "timestamp": 1627722468518, "user_tz": -330, "elapsed": 597, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5ab0c71f-d680-4419-dad2-602c2d84833a"
%%writefile /content/reco-tut-arr/code/data_ingestion.py
import os
import pandas as pd


# extract data from kaggle
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d calebemelike/akeed-restaurant

# unzip
!unzip akeed-restaurant.zip

# load into pandas dataframes
train_customers = pd.read_csv('train_customers.csv')
test_customers = pd.read_csv('test_customers.csv')
train_locations = pd.read_csv('train_locations.csv')
test_locations = pd.read_csv('test_locations.csv')
orders = pd.read_csv('orders.csv')
vendors = pd.read_csv('vendors.csv')

# schema correction for mixed datatypes
orders.iloc[:,[15,16,18,19,20]] = orders.iloc[:,[15,16,18,19,20]].astype('str')

# store data as bronze layer in parquet compressed format
data_path_bronze = "/content/reco-tut-arr/data/bronze/"

if not os.path.exists(data_path_bronze):
    os.makedirs(data_path_bronze)

def store_df(df,name):
    df.to_parquet(os.path.join(data_path_bronze,name+'.parquet.gz'), compression='gzip')

store_df(train_customers, 'train_customers')
store_df(test_customers, 'test_customers')
store_df(train_locations, 'train_locations')
store_df(test_locations, 'test_locations')
store_df(orders, 'orders')
store_df(vendors, 'vendors')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Mu3KGD52ukun" executionInfo={"status": "ok", "timestamp": 1627722504706, "user_tz": -330, "elapsed": 609, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="397f9aab-46ba-4f55-d1de-b61a0cea835c"
!cd /content/reco-tut-arr && git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZlDKEXfq00OO" executionInfo={"status": "ok", "timestamp": 1627722594908, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="75643b5a-c26c-4456-a407-110c661639eb"
!cd /content/reco-tut-arr && git checkout main
```

```python colab={"base_uri": "https://localhost:8080/"} id="KLV6t59d0lRc" executionInfo={"status": "ok", "timestamp": 1627722549451, "user_tz": -330, "elapsed": 622, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="97a76359-bc0c-4afe-de4c-77500873a434"
!cd /content/reco-tut-arr && git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="oZ-KUK_W0xrw" executionInfo={"status": "ok", "timestamp": 1627722727314, "user_tz": -330, "elapsed": 3068, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b23c7e1c-7415-4474-f141-85a6f35cfd5b"
!cd /content/reco-tut-arr && git push origin master
```
