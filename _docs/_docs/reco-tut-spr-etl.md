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

<!-- #region id="FZmKYvJbTWrq" -->
## Extract data from Kaggle
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fUZjO70FTZqz" executionInfo={"status": "ok", "timestamp": 1627713843682, "user_tz": -330, "elapsed": 10535, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="063320b0-4a83-455c-e0f2-33a23b4e526d"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c santander-product-recommendation
```

```python colab={"base_uri": "https://localhost:8080/"} id="IPzUOgGgTgY2" executionInfo={"status": "ok", "timestamp": 1627713875731, "user_tz": -330, "elapsed": 2400, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a7d11118-d8b2-4e5c-eacd-d157aa330a8e"
!unzip santander-product-recommendation.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="n_MWrdcbTqLy" executionInfo={"status": "ok", "timestamp": 1627713994372, "user_tz": -330, "elapsed": 29373, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="acce24c8-9773-484d-9a63-46fedee5d438"
!unzip -o sample_submission.csv.zip
!unzip -o test_ver2.csv.zip
!unzip -o train_ver2.csv.zip

!rm sample_submission.csv.zip
!rm test_ver2.csv.zip
!rm train_ver2.csv.zip
```

<!-- #region id="KbhZhN0zUQbR" -->
## Transform
<!-- #endregion -->

```python id="N69zfCnKULi_" executionInfo={"status": "ok", "timestamp": 1627714037862, "user_tz": -330, "elapsed": 410, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="GLdM-AclUSOB" executionInfo={"status": "ok", "timestamp": 1627714146957, "user_tz": -330, "elapsed": 401, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="189fe3f0-cd6d-4db5-923f-94990bf2dd02"
# testing the waters
df_train_sample = pd.read_csv('train_ver2.csv', nrows=5)
df_train_sample
```

```python colab={"base_uri": "https://localhost:8080/"} id="rORtHnTAUaMx" executionInfo={"status": "ok", "timestamp": 1627714412997, "user_tz": -330, "elapsed": 58955, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4a6866d0-1ad6-4dc2-ec5f-1b0210b92a91"
# loading full dataset
df_train = pd.read_csv('train_ver2.csv')
df_train.shape
```

<!-- #region id="I_57GFL4VwDE" -->
This warning is important. This kind of information will be useful to correctly save our dataframe in parquet format.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HWDb0ombUr1e" executionInfo={"status": "ok", "timestamp": 1627714414046, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="39f369a0-c84d-436c-fe9d-8bf577d8c5b4"
df_train.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="_7UZ4w4SUvHU" executionInfo={"status": "ok", "timestamp": 1627714741268, "user_tz": -330, "elapsed": 1033, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a62bae65-dcbe-4f40-81a1-ab08c94187d5"
# let's take 5th index column - in which we got mixed-dtype warning
df_train.iloc[:,5].unique()
```

<!-- #region id="B-qAlu4zWyGM" -->
It is all integers but apparantly some of them have whitespace as prefix. We can convert this column as int8 but this kind of wrangling and cleaning process we will do later in cleaning part. Right now, out focus is mainly ETL. So the best way is to explicitly convert these 4 columns into str dtype.
<!-- #endregion -->

```python id="2kdJhpFTYWCV" executionInfo={"status": "ok", "timestamp": 1627715427429, "user_tz": -330, "elapsed": 18057, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df_train.iloc[:,[5,8,11,15]] = df_train.iloc[:,[5,8,11,15]].astype('str')
```

```python id="YvPMj4KbWcmL" executionInfo={"status": "ok", "timestamp": 1627715516714, "user_tz": -330, "elapsed": 60298, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# now we can save it as parquet
df_train.to_parquet('df_train.parquet.gzip', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Zx9DMTyvXxDi" executionInfo={"status": "ok", "timestamp": 1627715562262, "user_tz": -330, "elapsed": 2280, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bb74bbfe-6e73-4498-aa10-1599c347b98c"
# let's clean the memeory for our next dataset
import gc
del df_train
gc.collect()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="rGdHfRo8aF7w" executionInfo={"status": "ok", "timestamp": 1627715602505, "user_tz": -330, "elapsed": 404, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a4c51509-5833-4e33-f6c2-345af1725abb"
# now let's load test set
# testing the waters
df_test_sample = pd.read_csv('test_ver2.csv', nrows=5)
df_test_sample
```

```python colab={"base_uri": "https://localhost:8080/"} id="NtK_hheuaQQp" executionInfo={"status": "ok", "timestamp": 1627715635015, "user_tz": -330, "elapsed": 4129, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="102cdd31-2386-4a44-fb6e-10077723a4c0"
# loading full dataset
df_test = pd.read_csv('test_ver2.csv')
df_test.shape
```

```python id="21kn9T0kaXRd" executionInfo={"status": "ok", "timestamp": 1627715674569, "user_tz": -330, "elapsed": 390, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df_test.iloc[:,[15]] = df_test.iloc[:,[15]].astype('str')
```

```python id="-AYhsBfwah3P" executionInfo={"status": "ok", "timestamp": 1627715697420, "user_tz": -330, "elapsed": 3787, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df_test.to_parquet('df_test.parquet.gzip', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/"} id="6nPZTn2gawr_" executionInfo={"status": "ok", "timestamp": 1627715743072, "user_tz": -330, "elapsed": 377, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc41e40e-f5e7-4946-a6b9-00379611a473"
del df_test
gc.collect()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="vjYwEzpzaoty" executionInfo={"status": "ok", "timestamp": 1627715760172, "user_tz": -330, "elapsed": 375, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8104430a-6335-4692-f40c-a18bed21b3ee"
df = pd.read_csv('sample_submission.csv')
df.head()
```

```python id="C-9dH-aSarjs" executionInfo={"status": "ok", "timestamp": 1627715800784, "user_tz": -330, "elapsed": 1066, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df.to_parquet('df_submission.parquet.gzip', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/"} id="2HwrzE0XasFF" executionInfo={"status": "ok", "timestamp": 1627716274819, "user_tz": -330, "elapsed": 610, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f3f095d4-52f4-4053-bf40-20eb9ef09350"
from os.path import getsize as gs

print('train set size reduced: {:.2f} MB -> {:.2f} MB'.format(gs('train_ver2.csv')/1e6, gs('df_train.parquet.gzip')/1e6))
print('test set size reduced: {:.2f} MB -> {:.2f} MB'.format(gs('test_ver2.csv')/1e6, gs('df_test.parquet.gzip')/1e6))
print('submission set size reduced: {:.2f} MB -> {:.2f} MB'.format(gs('sample_submission.csv')/1e6, gs('df_submission.parquet.gzip')/1e6))
```

<!-- #region id="rVxGtmjDbUub" -->
## Store these files into cloud
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="B4WJ6k3ic90x" executionInfo={"status": "ok", "timestamp": 1627716404196, "user_tz": -330, "elapsed": 82256, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6e989f1f-17ab-4b86-b6eb-4f8dfc9d17fb"
!wget -O YANDEX-DISK-KEY.GPG http://repo.yandex.ru/yandex-disk/YANDEX-DISK-KEY.GPG
!apt-key add YANDEX-DISK-KEY.GPG
!echo "deb http://repo.yandex.ru/yandex-disk/deb/ stable main" >> /etc/apt/sources.list.d/yandex-disk.list
!apt-get update
!apt-get install yandex-disk
!yandex-disk setup
```

```python id="_T-rFPbUdZV5" executionInfo={"status": "ok", "timestamp": 1627716554868, "user_tz": -330, "elapsed": 1091, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!cd /content/recodata && mkdir -p santander/v1
!mv ./*.parquet.gzip /content/recodata/santander/v1
```

<!-- #region id="8Hv3I7AJekVj" -->
## Registry as api in recochef
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DIArYZYJeQs9" executionInfo={"status": "ok", "timestamp": 1627716840267, "user_tz": -330, "elapsed": 2879, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="37763912-eeac-4691-c8a2-7e26644efcde"
project_name = "recochef"; branch = "master"; account = "sparsh-ai"
!cp /content/drive/MyDrive/mykeys.py /content
import sys; sys.path.append("/content/drive/MyDrive"); import mykeys

path = "/content/dev/" + project_name
!mkdir -p "{path}"
%cd "{path}"
sys.path.append(path)

!git config --global user.email "chef@recohut.com"
!git config --global user.name  "recochef-dev"
!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
%cd /content
```

```python colab={"base_uri": "https://localhost:8080/"} id="oBaY9WZngFFT" executionInfo={"status": "ok", "timestamp": 1627717158477, "user_tz": -330, "elapsed": 18665, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b1df53e5-121e-4f78-ea42-6f7e45c1a194"
!pip install -U -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="v9_y7szze_bt" executionInfo={"status": "ok", "timestamp": 1627717349662, "user_tz": -330, "elapsed": 390, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# get shared links from drive and build data class

train_url = 'https://disk.yandex.ru/d/e0OhgI-xB13UPQ'
test_url = 'https://disk.yandex.ru/d/yC5WeXLIyNOV2g'
submission_url = 'https://disk.yandex.ru/d/HRVCqhypKtZcZQ'
```

```python id="644FDS16fxLt" executionInfo={"status": "ok", "timestamp": 1627717374584, "user_tz": -330, "elapsed": 1058, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from data_cache import pandas_cache
import pandas as pd
import pickle
import os

from recochef.datasets.dataset import Dataset
from recochef.utils._utils import download_yandex


class Santander(Dataset):
  def __init__(self, version='v1'):
    super(Santander, self).__init__()
    self.version = version

  @pandas_cache
  def load_train(self, filepath='train.parquet.gz'):
    # fileurl = self.permalinks['santander'][self.version]['train']
    fileurl = train_url
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    train = pd.read_parquet(filepath)
    return train

  @pandas_cache
  def load_test(self, filepath='test.parquet.gz'):
    # fileurl = self.permalinks['santander'][self.version]['test']
    fileurl = test_url
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    test = pd.read_parquet(filepath)
    return test

  @pandas_cache
  def load_submission(self, filepath='submission.parquet.gz'):
    # fileurl = self.permalinks['santander'][self.version]['submission']
    fileurl = submission_url
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    submission = pd.read_parquet(filepath)
    return submission
```

```python id="3kjYhMI8hAvP" executionInfo={"status": "ok", "timestamp": 1627717401953, "user_tz": -330, "elapsed": 630, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
sdata = Santander()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="XqkAoF04hHht" executionInfo={"status": "ok", "timestamp": 1627717531829, "user_tz": -330, "elapsed": 107848, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e076d51f-b97b-4765-a895-38b313a31145"
df_train = sdata.load_train()
df_train.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="iL5WV9eYhNDZ" executionInfo={"status": "ok", "timestamp": 1627717533289, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dd7336fd-ed6b-425f-b24f-0e09fca9baef"
df_train.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="yxK7jUWKhQBD" executionInfo={"status": "ok", "timestamp": 1627717626614, "user_tz": -330, "elapsed": 7535, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6620444b-cb02-44da-dc15-68242daa8c86"
df_test = sdata.load_test()
df_test.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="petdqFPvh8p0" executionInfo={"status": "ok", "timestamp": 1627717655252, "user_tz": -330, "elapsed": 4262, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eaf07123-813a-4942-d14b-7c03bdec4134"
df_sub = sdata.load_submission()
df_sub.head()
```

<!-- #region id="lEOgYIENiIAN" -->
Finally, let's save this data class in recochef, and we are done!
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="IAeuD3yNiPc4" executionInfo={"status": "ok", "timestamp": 1627717742665, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ea977e86-d0e8-47bd-8562-1d2425943c44"
%%writefile /content/dev/recochef/src/recochef/datasets/santander.py
from data_cache import pandas_cache
import pandas as pd
import pickle
import os

from recochef.datasets.dataset import Dataset
from recochef.utils._utils import download_yandex


class Santander(Dataset):
  def __init__(self, version='v1'):
    super(Santander, self).__init__()
    self.version = version

  @pandas_cache
  def load_train(self, filepath='train.parquet.gz'):
    fileurl = self.permalinks['santander'][self.version]['train']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    train = pd.read_parquet(filepath)
    return train

  @pandas_cache
  def load_test(self, filepath='test.parquet.gz'):
    fileurl = self.permalinks['santander'][self.version]['test']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    test = pd.read_parquet(filepath)
    return test

  @pandas_cache
  def load_submission(self, filepath='submission.parquet.gz'):
    fileurl = self.permalinks['santander'][self.version]['submission']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    submission = pd.read_parquet(filepath)
    return submission
```

```python colab={"base_uri": "https://localhost:8080/"} id="O2Re19JBiauk" executionInfo={"status": "ok", "timestamp": 1627717772511, "user_tz": -330, "elapsed": 1014, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ed988dd5-3a6f-431c-fee9-b7c72a619cc9"
!cd /content/dev/recochef && git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="zEx3PBu5idFZ" executionInfo={"status": "ok", "timestamp": 1627717847202, "user_tz": -330, "elapsed": 3475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="908044b8-1072-4d1c-ba7b-74c03770d694"
!cd /content/dev/recochef && git add . && git commit -m 'added Santander dataset' && git push origin master
```
