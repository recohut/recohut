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

<!-- #region id="hWCH9YIysyZn" -->
## Environment Setup
<!-- #endregion -->

```python id="Z3qjPp055tXf" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1644214755663, "user_tz": -330, "elapsed": 15865, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0205ad81-18be-408c-8a96-be626c56eb59"
project_name = 'daly_nlp'; branch = 'main'; account = 'sparsh-ai'
import os
project_path = os.path.join('/content', branch)
if not os.path.exists(project_path):
    !apt-get -qq install tree
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
else:
    %cd "{project_path}"
```

```python id="7lAKlgUD5tXi"
!cd /content/"{branch}" && git add .
!cd /content/"{branch}" && git commit -m 'commit'
!cd /content/"{branch}" && git pull --rebase origin "{branch}"
!cd /content/"{branch}" && git push origin "{branch}"
```

<!-- #region id="RE0sdHucs4rQ" -->
## Code
<!-- #endregion -->

```python id="lIYdn1woOS1n"
import pandas as pd
import requests
from tqdm import tqdm
```

```python id="WkDhG_XNrqTz"
url_link = f'https://finance.yahoo.com/screener/unsaved/250a16f6-3cda-4062-8d6a-0d17d2ce4867?count=100&offset={i}'
r = requests.get(url_link,headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
read_html_pandas_data = pd.read_html(r.text)
data.append(pd.DataFrame(read_html_pandas_data[0])[['Symbol','Name']])
```

```python id="piO7BUBck-Dg"
data = []

for i in tqdm(range(0,70000,100)):
    try:
        url_link = f'https://finance.yahoo.com/screener/unsaved/250a16f6-3cda-4062-8d6a-0d17d2ce4867?count=100&offset={i}'
        r = requests.get(url_link,headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        read_html_pandas_data = pd.read_html(r.text)
        data.append(pd.DataFrame(read_html_pandas_data[0])[['Symbol','Name']])
    except:
        print(f'Skipped at i={i}')
```

```python id="Ar_ghxW2qv0c" executionInfo={"status": "ok", "timestamp": 1644211944625, "user_tz": -330, "elapsed": 429, "user": {"displayName": "", "photoUrl": "", "userId": ""}} outputId="42c9bb91-b14d-445c-a2c2-e529c7172cff" colab={"base_uri": "https://localhost:8080/"}
len(data)
```

```python id="4i_b9UnTqwf8"
pd.concat(data, axis=0).to_excel('./data/processed/yahoo_finance_stocklist.xlsx')
```

<!-- #region id="GM1jMQww1pyn" -->
---
<!-- #endregion -->

```python id="cV1DMILC1TFc" executionInfo={"status": "ok", "timestamp": 1644214780092, "user_tz": -330, "elapsed": 779, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="Cji-WiHk1no9" executionInfo={"status": "ok", "timestamp": 1644214801134, "user_tz": -330, "elapsed": 1614, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f1f0e867-ca2b-4cc5-9267-9de02a2ad830"
pd.read_excel('/content/main/data/processed/yahoo_finance_stocklist.xlsx', index_col=[0])
```

```python id="qnFswfoP1pHx"

```
