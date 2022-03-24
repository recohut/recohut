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

```python id="rm3pirta8J3X" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1601057868062, "user_tz": -330, "elapsed": 2435, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import json
import pandas as pd 
```

```python id="ypZr8W3Nf3CA" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1601059369514, "user_tz": -330, "elapsed": 20663, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
x = pd.read_json('products.json')
dict_cols = ['attributes','Name','Description','Text','Keywords','MainCharacteristics']
for colname in dict_cols:
  x[colname] = x[colname].apply(lambda x: {} if pd.isna(x) else x)
  _tempdf = pd.json_normalize(x[colname], errors='ignore')
  _tempdf = _tempdf.add_prefix(colname+'_')
  x = pd.concat([x, _tempdf], axis=1, sort=False)
x = x.drop(dict_cols, axis=1)
```

```python id="NYlzTu4tgTlF" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 734} executionInfo={"status": "ok", "timestamp": 1601059388616, "user_tz": -330, "elapsed": 34505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b5b071e6-0d26-4f4b-ecdd-fd62588c4568"
x.sample(10)
```

```python id="Mp-PKWCFlqHm" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 294} executionInfo={"status": "ok", "timestamp": 1601059604049, "user_tz": -330, "elapsed": 6775, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c1de36b6-68ef-40e9-f376-23b9fd62c3ba"
!pip install clean-text[gpl]
```

```python id="b6GEb_nZlMn9" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 86} executionInfo={"status": "ok", "timestamp": 1601059631214, "user_tz": -330, "elapsed": 3504, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f45276fa-fd7b-4d7b-d47a-12e8a07bc1d1"
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import groupby
from bs4 import BeautifulSoup
from cleantext import clean
from collections import OrderedDict

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
stopwords = list(set(stopwords.words('english')))

ps = PorterStemmer()
nltk.download('wordnet') 
lemmatizer = WordNetLemmatizer()
```

```python id="CkfLX05SmMlf" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1601059859620, "user_tz": -330, "elapsed": 995, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def clean_text(text):
  return clean(text,
                fix_unicode=True,               # fix various unicode errors
                to_ascii=True,                  # transliterate to closest ASCII representation
                lower=False,                     # lowercase text
                no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
                no_urls=True,                  # replace all URLs with a special token
                no_emails=True,                # replace all email addresses with a special token
                no_phone_numbers=True,         # replace all phone numbers with a special token
                no_numbers=False,               # replace all numbers with a special token
                no_digits=False,                # replace all digits with a special token
                no_currency_symbols=True,      # replace all currency symbols with a special token
                no_punct=False,                 # fully remove punctuation
                replace_with_url=' url ',
                replace_with_email=' email ',
                replace_with_phone_number=' phone ',
                lang='en'
            )
```

```python id="Vc1GHS30liCi" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1601059873681, "user_tz": -330, "elapsed": 1010, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def clean_l1(text):
  text = ' ' + text + ' '
  text = BeautifulSoup(text, "lxml").text
  text = re.sub(r'<.*?>', ' ', text)
  text = re.sub(r'\{[^{}]*\}', ' ', text)
  text = BeautifulSoup(text, "lxml").text
  text = clean_text(text)
  text = re.sub(r'[^A-Za-z0-9.,?\']', ' ', text)
  text = re.sub('[.]', ' . ', text)
  text = re.sub('[,]', ' , ', text)
  text = re.sub('[?]', ' ? ', text)
  text = ' '.join(text.split())
  text = re.sub(r'\b\w{1,1}\b', '', text)
  text = ' '.join([k for k,v in groupby(text.split())])  
  text = ' '.join(text.split())
  return text
```

```python id="YNjChDh4mHch" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1601059985556, "user_tz": -330, "elapsed": 8989, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
x['Text_en'] = x['Text_en'].astype('str').apply(clean_l1)
x['Description_en'] = x['Description_en'].astype('str').apply(clean_l1)
```

```python id="nu6K8QcnmTmS" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1601060068914, "user_tz": -330, "elapsed": 1042, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
x.sample(100).to_csv('sample_output.csv', index=False)
```

```python id="vNUryKYGnLQG" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1601060122619, "user_tz": -330, "elapsed": 1147, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mv /content/*.json '/content/drive/My Drive/Upwork/Temp'
```

```python id="H1C-w5Hpnqqu" colab_type="code" colab={}

```
