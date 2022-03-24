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

<!-- #region id="Zr5YJ5xSZqis" -->
# Email Classification
<!-- #endregion -->

<!-- #region id="-npoQSyFTTHM" -->
## Fetching data from MS-Sql
<!-- #endregion -->

```python id="X3AtNr2SaSCa"
!apt install unixodbc-dev
!pip install pyodbc
```

```sh id="BLRNUZ-Pb4iT"
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get -q -y install msodbcsql17
```

```python id="nGEqTyX5a7w2"
import os
import pyodbc
import urllib
import pandas as pd
from sqlalchemy import create_engine
```

```python id="jVXQEhu4bJt5"
driver = [item for item in pyodbc.drivers()][-1]
conn_string = f'Driver={driver};Server=tcp:server.<domain>.com,<port>;Database=<db>;Uid=<userid>;Pwd=<pass>;Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;'
conn = pyodbc.connect(conn_string)
cursor = conn.cursor()
```

```python id="MPC2RpHco7C-"
# params = urllib.parse.quote_plus(conn_string)
# conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
# engine_feat = create_engine(conn_str, echo=True)
# print(engine_feat.table_names())
```

```python id="CQN-9vldsTgA" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1597254541382, "user_tz": -330, "elapsed": 1563, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="831cb0d5-2123-47bc-aa7f-b2f85b2537ea"
tname = 'tbl_Final_Lable_Data_18_n_19'
query = f'select count(*) from {tname}'

cursor.execute(query)
cursor.fetchall()
```

```python id="baJZZ5ww2eZm" colab={"base_uri": "https://localhost:8080/", "height": 442} executionInfo={"status": "ok", "timestamp": 1597254977341, "user_tz": -330, "elapsed": 1886, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="08a74fb4-4460-4c75-e632-426b820b36de"
query = f'select top 5 * from {tname}'
df = pd.read_sql(query, conn)
df.info()
```

```python id="6Hy570Spsev7"
%reload_ext google.colab.data_table
```

```python id="IsOsExH-txzH"
df
```

```python id="U9uhMDtV1GDk" colab={"base_uri": "https://localhost:8080/", "height": 119} executionInfo={"status": "ok", "timestamp": 1597255219329, "user_tz": -330, "elapsed": 1603, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8a63c900-b40d-4bfc-8061-26eedcbb45d5"
df.columns
```

```python id="C09UEiqUt35P" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1597262681717, "user_tz": -330, "elapsed": 6654848, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="38143115-810a-4095-d27b-ed6aae238fb0"
query = f'select tSubject, mMsgContent, QueryType, SubQueryType from {tname}'
df = pd.read_sql(query, conn)
df.info()
```

```python id="thdlJgVO1Gxy"
df.to_pickle('data.p')
```

<!-- #region id="8q8YbNDWTqo4" -->
## Wrangling
<!-- #endregion -->

```python id="Lcbzjg0pCCyh"
# wrangling.py
import os
import numpy as np
import pandas as pd
spath = '/content/email_class'
df = pd.read_pickle(os.path.join(spath,'data','raw','data.p'))
df.columns = ['subj','msg','qtype','stype']
df['type'] = df['qtype'] + ' | ' + df['stype']
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(how='all')
df = df.drop_duplicates()
df = df.dropna(subset=['subj', 'msg'], how='all')
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(how='all')
df = df.drop_duplicates()
df = df.dropna(subset=['subj', 'msg'], how='all')
df = df.fillna(' ')
df['subj&msg'] = df['subj'] + ' sub_eos_token ' + df['msg']
df = df[['subj&msg','type']]
df.columns = ['text','target']
df.sample(10000).to_pickle('df_raw_wrangled_sample_10k.p')
# df.sample(10000).to_pickle(os.path.join(spath,'data','wrangled','df_raw_wrangled_sample_10k.p'))
# df.to_pickle(os.path.join(spath,'data','wrangled','df_raw_wrangled_full.p'))
```

```python id="hlb8-lExVmiw"
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

tqdm.pandas()
%reload_ext autoreload
%autoreload 2
%reload_ext google.colab.data_table
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')
```

```python id="w79QgqP_23__" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1597560238130, "user_tz": -330, "elapsed": 42402, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="15e2bc71-2d04-4454-fbf0-176b80bff19e"
df = pd.read_pickle(os.path.join(spath,'data','raw','data.p'))
df.info()
```

```python id="7XK3KUYZ34tZ"
df.sample(20)
```

```python id="wYm1-URg7C66"
df.columns
```

```python id="gj2zOluv7wPt"
df.columns = ['subj','msg','qtype','stype']
```

```python id="2n7iU3FO8AXW"
df.qtype.nunique()
```

```python id="o5tEpnms8KOY"
df.qtype.value_counts()[:50]
```

```python id="Hfqmljfx8bVB" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1597341029592, "user_tz": -330, "elapsed": 1346, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="070d2412-b788-4ed7-fea6-9a3e6e5afea3"
df.stype.nunique()
```

```python id="YdF-08448Peu"
df.stype.value_counts()[:50]
```

```python id="D5SalQHJz2yW"
df['type'] = df['qtype'] + ' | ' + df['stype']
```

```python id="6gMwjYmp0C7W" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1597406059662, "user_tz": -330, "elapsed": 902, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4bed2b4b-a399-482d-f714-2fb719ad8ee4"
df['type'].nunique()
```

```python id="V1WutEGB0PJf"
df['type'].value_counts()[:50]
```

```python id="TRGQM5TR9Qq-" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1597341249538, "user_tz": -330, "elapsed": 1786, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3a106282-634e-4ba7-fcd4-bb4892bc3d04"
df.subj.nunique()
```

```python id="t8mJgC5d8WBq"
df.subj.value_counts()[:50]
```

```python id="kSKsChwu9euw" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1597341313014, "user_tz": -330, "elapsed": 4615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eaaee5f2-4680-47ad-b9e1-7f0051dab6f9"
df.msg.nunique()
```

```python id="Ik-uzC1X9B-B"
df.msg.value_counts()[:50]
```

```python id="uEA7-SWw9ZMg"
df[df.msg.isnull()].sample(10)
```

```python id="j0Dg_-ii-CdT" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1597341729282, "user_tz": -330, "elapsed": 1832, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2caf82a9-e0df-497d-ace5-b68fa7ca75ba"
df[(df.msg.isnull()) & (df.subj.isnull())].info()
```

```python id="a4h8ahQ1-v23" colab={"base_uri": "https://localhost:8080/", "height": 221} executionInfo={"status": "ok", "timestamp": 1597560296004, "user_tz": -330, "elapsed": 38247, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2ff96fdc-341a-4976-89e7-06f16f279ab4"
df2 = df.replace(r'^\s*$', np.nan, regex=True)
df2.info()
```

```python id="0jrw87Xq_ftb" colab={"base_uri": "https://localhost:8080/", "height": 221} executionInfo={"status": "ok", "timestamp": 1597560297220, "user_tz": -330, "elapsed": 12766, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f7b8bfa3-f52e-4ce1-f2dd-75ea8960b797"
df3 = df2.dropna(how='all')
df3.info()
```

```python id="SKOlzfUNAbXP" colab={"base_uri": "https://localhost:8080/", "height": 221} executionInfo={"status": "ok", "timestamp": 1597560302536, "user_tz": -330, "elapsed": 14100, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a9526948-0b12-4312-c34b-4577e8cc3286"
df4 = df3.drop_duplicates()
df4.info()
```

```python id="OuRnQAbyAlPB"
df4[(df4.msg.isnull()) & (df4.subj.isnull())]
```

```python id="uhzvQ4tHBWQh" colab={"base_uri": "https://localhost:8080/", "height": 221} executionInfo={"status": "ok", "timestamp": 1597560302539, "user_tz": -330, "elapsed": 5413, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7afb675a-2a10-43fb-f1fe-5a4bf657325a"
df5 = df4.dropna(subset=['subj', 'msg'], how='all')
df5.info()
```

```python id="lRdLXC_XCBv_" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1597560308040, "user_tz": -330, "elapsed": 1665, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a2c8c1f7-68bb-4157-a266-77f3bee8b906"
df4.shape, df5.shape
```

```python id="_XgU0TDhCMYR"
sample = df5.sample(10)
sample
```

```python id="X1QhUkCUYRA9"
!pip install ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
text_processor = TextPreProcessor(
  normalize=['url', 'email', 'percent', 'money', 'phone', 
              'user', 'time', 'date', 'number'],
  # annotate={"hashtag", "allcaps", "elongated", "repeated",
  #           'emphasis', 'censored'},
  fix_html=True,
  segmenter="twitter",
  corrector="twitter", 
  unpack_hashtags=True,
  unpack_contractions=True,
  spell_correct_elong=False,
  tokenizer=SocialTokenizer(lowercase=False).tokenize,
  dicts=[emoticons]
  )
```

```python id="ZXHVLF4f1lSB"
import re
from bs4 import BeautifulSoup
```

```python id="B8AeiZmRGy9M"
# text_raw = sample.loc[[299500]].msg.tolist()[0]
# text = text_raw
# text = BeautifulSoup(text, "lxml").text
# text = re.sub(r'<.*?>', ' ', text)
# text = re.sub(r'\{[^{}]*\}', ' ', text)
# text = re.sub(r'\s', ' ', text)
# text = re.sub(r'.*\..*ID.*?(?=\s)', ' ', text)
# text = re.sub(r'DIV..*?(?=\s)', ' ', text)
# text = BeautifulSoup(text, "lxml").text
# text = text.strip()
# text = " ".join(text_processor.pre_process_doc(text))
# text
```

```python id="s2aAG-BXEHSg"
from itertools import groupby
```

```python id="RhH8AaG4reu8"
html_residual = 'P . ImprintUniqueID LI . ImprintUniqueID DIV . ImprintUniqueID TABLE . ImprintUniqueIDTable DIV . Section '
caution_residual =  'CAUTION This email originated from outside of the organization . Do not click links or open attachments unless you recognize the sender and know the content is safe . '
```

```python id="875RLvXEZfwj"
def clean_text(text):
  text = ' ' + text + ' '
  text = BeautifulSoup(text, "lxml").text
  text = re.sub(r'<.*?>', ' ', text)
  text = re.sub(r'\{[^{}]*\}', ' ', text)
  text = re.sub(r'\s', ' ', text)
  # text = re.sub(r'(?=\s).*\..*ID.*?(?=\s)', ' ', text)
  # text = re.sub(r'(?=\s)DIV..*?(?=\s)', ' ', text)
  text = re.sub(r'Forwarded message.*?(?=____)', ' ', text)
  text = BeautifulSoup(text, "lxml").text
  text = ' '.join(text_processor.pre_process_doc(text))
  text = re.sub(r'[^A-Za-z<>. ]', ' ', text)
  text = ' '.join(text.split())
  text = re.sub(html_residual, '', text)
  text = re.sub(caution_residual, '', text)
  text = re.sub(r'(?:\d+[a-zA-Z]+|[a-zA-Z]+\d+)', '<hash>', text)
  # text = re.sub(r'\b\w{1,2}\b', '', text)
  text = ' '.join(text.split())
  text = ' '.join([k for k,v in groupby(text.split())])
  return text
```

```python id="ndCIDKuek8nj"
# # text_raw = sample.loc[[75806]].msg.tolist()[0]
# text = text_raw
# text = BeautifulSoup(text, "lxml").text
# text = re.sub(r'<.*?>', ' ', text)
# text = re.sub(r'\{[^{}]*\}', ' ', text)
# text = re.sub(r'\s', ' ', text)
# # text = re.sub(r'.*\..*ID.*?(?=\s)', ' ', text)
# # text = re.sub(r'DIV..*?(?=\s)', ' ', text)
# text = re.sub(r'Forwarded message.*?(?=____)', ' ', text)
# text = BeautifulSoup(text, "lxml").text
# text = " ".join(text_processor.pre_process_doc(text))
# text = re.sub(r'[^A-Za-z0-9<>. ]', ' ', text)
# text = ' '.join(text.split())
# text
```

```python id="ZNWWpQLGjfJB"
sample = df5.sample(1000)
```

```python id="doOt0vI3vnsq"
sample['subj_clean'] = sample['subj'].fillna(' ').apply(clean_text)
```

```python id="A_R-SBxY2aST"
[(x,y) for x,y in zip(sample.subj.tolist()[:50],sample.subj_clean.tolist()[:50])]
```

```python id="QB_k4RX7auVr"
sample['msg_clean'] = sample['msg'].fillna(' ').apply(clean_text)
sample.msg.tolist()
```

```python id="gyp01dH0hwsL"
sample.msg_clean.tolist()
```

```python id="ZOq1zBOj4F8S" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1597410713287, "user_tz": -330, "elapsed": 7056, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6f75514a-fec3-408a-eae5-8fa6f454c7ea"
sample = df5.sample(1000, random_state=40)
sample['subj_clean'] = sample['subj'].fillna(' ').apply(clean_text)
sample['msg_clean'] = sample['msg'].fillna(' ').apply(clean_text)
sample['subj&msg'] = sample['subj_clean'] + ' | ' + sample['msg_clean']
sample = sample[['subj&msg','type']]
sample.columns = ['text','target']
sample.info()
```

```python id="vDePhF264w__"
sample
```

```python id="v7RXJnql5aAX" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1597426275158, "user_tz": -330, "elapsed": 4538335, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="381b5dac-8a76-4fdc-b21c-00bcf535f970"
sample = df5.copy()
sample['subj_clean'] = sample['subj'].fillna(' ').apply(clean_text)
sample['msg_clean'] = sample['msg'].fillna(' ').apply(clean_text)
sample['subj&msg'] = sample['subj_clean'] + ' | ' + sample['msg_clean']
sample = sample[['subj&msg','type']]
sample.columns = ['text','target']
sample.info()
```

```python id="TVVTXU7kFM3a" colab={"base_uri": "https://localhost:8080/", "height": 68} executionInfo={"status": "ok", "timestamp": 1597427222296, "user_tz": -330, "elapsed": 3555, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3d277c72-06cd-4deb-82fb-a482546d578d"
sample.nunique()
```

```python id="cAFQ27-hE8-X" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1597427243634, "user_tz": -330, "elapsed": 4552, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1f848bd6-6255-47d4-d999-c56df0e8caac"
sample2 = sample.drop_duplicates()
sample2.info()
```

```python id="DbvywuvOFhso" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1597427354490, "user_tz": -330, "elapsed": 11001, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="001d1e54-7f9b-4aea-a6a5-7ab3dfb544ec"
sample3 = sample2.replace(r'^\s*$', np.nan, regex=True)
sample3.info()
```

```python id="OXUDri7LB8JK"
df5.to_pickle('df_raw_wrangled.p')
```

```python id="Eq78wXKwF-7d" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1597427495497, "user_tz": -330, "elapsed": 1060, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="58727a8f-d9d7-4184-ccdc-739319a2d45f"
sample2.info()
```

```python id="31iN8-7PGSoh" colab={"base_uri": "https://localhost:8080/", "height": 68} executionInfo={"status": "ok", "timestamp": 1597427506045, "user_tz": -330, "elapsed": 3731, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ee4bfc45-f0b1-4202-e08a-ff5341e55ef2"
sample2.nunique()
```

```python id="gxFziBUVGUmh"
sample2.describe()
```

<!-- #region id="itwk9pEuGXU4" -->
## Text Cleaning
<!-- #endregion -->

```python id="TMAiI_G3sDew"
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = list(set(stopwords.words('english')))

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') 
lemmatizer = WordNetLemmatizer() 

from nltk.stem import PorterStemmer
ps = PorterStemmer()

import warnings
warnings.filterwarnings("ignore")

tqdm.pandas()
%reload_ext autoreload
%autoreload 2
%reload_ext google.colab.data_table
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')
```

```python id="nMk12EPzsd5B" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1597897404475, "user_tz": -330, "elapsed": 6977, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c27caa87-def8-49df-8994-985726270da3"
df = pd.read_pickle(os.path.join(path,'data','wrangled','df_raw_wrangled_sample_10k.p'))
df.info()
```

```python id="1gfvcYwqtVsJ"
df.sample(5)
```

```python id="JEFv-QYqyO_q"
df[df.target=='<redacted>'].sample(5)
```

```python id="Llthvl8wtYa2"
df.target.value_counts()
```

```python id="4pVEL8xlvVnp"
# # cleaning pipe
# - lowercase
# - remove nonalpha
# - stopword
# - lemmatization
# - stemming

# - min occurence
# - max occurence
# - ngram
# - misspell
# - contraction

# - encode plus tokenizer
```

```python id="a_rpa62rG6O0"
df = df.reset_index(drop=True)
```

```python id="Ot-con96Ia5V"
# label tokens
caution_label = 'CAUTION: This email originated from outside of the organization. \
Do not click links or open attachments unless you recognize the sender and know the \
content is safe'

confidential_label = '<redacted>'

confidential_label = '<redacted>'

retransmit_label = '<redacted>'

alert_label = '<redacted>'

html_labels = ['P.ImprintUniqueID', 'LI.ImprintUniqueID', 'DIV.ImprintUniqueID',
              'TABLE.ImprintUniqueIDTable', 'DIV.Section1']
html_regex = re.compile('|'.join(map(re.escape, html_labels)))

newline_token = '\n'


custom_stopwords = ['best', 'regard', 'direct', 'number', 'phone', 'mobile', 'number', 'reply', 'url', 'com']

```

```python id="8OkAUM4wYv9D"
!pip install clean-text
```

```python id="78M8BYE2J8dZ"
def clean_l1(text):
  text = re.sub(caution_label, ' cautionlabel ', text)
  text = re.sub(confidential_label, ' confidentiallabel ', text)
  text = html_regex.sub('htmltoken', text)
  text = re.sub(retransmit_label, ' retransmittoken ', text)
  text = re.sub(alert_label, ' alerttoken ', text)
  text = re.sub('sub_eos_token', ' bodytoken ', text)
  text = ' ' + text + ' '
  text = BeautifulSoup(text, "lxml").text
  text = re.sub(r'<.*?>', ' ', text)
  text = re.sub(r'\{[^{}]*\}', ' ', text)
  # # text = re.sub(r'Forwarded message.*?(?=____)', ' ', text)
  text = BeautifulSoup(text, "lxml").text
  text = re.sub(newline_token, ' newlinetoken ', text)
  text = ' '.join(text.split())
  text = re.sub(r'[^A-Za-z.,?\'@]', ' ', text)
  

  # text = ' '.join(text.split())
  return text
```

```python id="4DnIxxwPIzM5"
xx = clean_l1(df.loc[idx,'text']); xx
# print(xx)
```

```python id="PhPlfTu5J4ar"
df.loc[idx,'text']
```

```python id="Ju5aimGEHMOD"
idx = np.random.randint(0,len(df))
print(idx)
print(df.loc[idx,'text'])
```

```python id="Q_VsdTXgKcf7"
# idx = 9
# print(df.text.iloc[[idx]].tolist()[0])
# xx = df.text.iloc[[idx]].apply(clean_l1).tolist()[0]
# xx
```

```python id="rZyxmG3DWDl9"
df['text_clean_l1'] = df.text.apply(clean_l1)
```

```python id="nE6_ASAAWNxw"
df.text_clean_l1.sample().tolist()[0]
```

```python id="cjkMbFtLIOfJ"
set1_words = ['best regards', 'regards', 'thanks regards', 'warm regards']
set1_regex = re.compile('|'.join(map(re.escape, set1_words)))
```

```python id="6bGL2Mrbknd6"
from itertools import groupby
```

```python id="5oGtFkkWLBui"
def replace_words(s, words):
  for k, v in words.items():
      s = s.replace(k, v)
  return s

word_mapping = {' f o ':' fno ',
                ' a c ':' account ',
                ' a/c ':' account ',
                ' fw ':' forward ',
                ' fwd ':' forward ',
                ' forwarded ':' forward ',
                ' no. ':' number ',
                }
```

```python id="kbpDvIA-FhfB"
def clean_l2(text):
  text = ' ' + text + ' '
  text = text.lower()
  text = ' '.join(text.split())
  text = replace_words(text, word_mapping)
  text = re.sub('[.]', ' . ', text)
  text = re.sub('[,]', ' , ', text)
  text = re.sub('[?]', ' ? ', text)
  text = ' '.join([w for w in text.split() if re.match('^[a-z.,?\'\-\~#`!&*()]+$', w)])
  text = re.sub(r'[^a-z.,?\']', ' ', text)
  text = set1_regex.sub('eostoken', text)
  text = text + ' eostoken '
  text = re.match(r'^.*?eostoken', text).group(0)
  text = re.sub(r'eostoken', '', text)
  text = re.sub(r'\b\w{1,1}\b', '', text)
  text = ' '.join([k for k,v in groupby(text.split())])
  text = ' '.join(text.split())
  return text
```

```python id="Iy6WBvlvzQRg" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1597590978429, "user_tz": -330, "elapsed": 1714, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="69b1a426-0b62-4f4c-861b-85a408d80bca"
xxy = 'warm regards regrads hello regards'
xxy = ' '.join([w for w in xxy.split() if re.match('^[a-z.,?\'\-\~#`!&*()]+$', w)])
xxy = re.sub(r'[^a-z.,?\']', ' ', xxy)
xxy = set1_regex.sub('eostoken', xxy)
re.match(r'^.*?eostoken', xxy).group(0)
```

```python id="3Nv6FemDHsvM"
idx = 1
# print(df.text.iloc[[idx]].tolist()[0])
print(df.text_clean_l1.iloc[[idx]].tolist()[0])
df.text_clean_l1.iloc[[idx]].apply(clean_l2).tolist()[0]
```

```python id="KJPVTdKXn7tZ"
df['text_clean_l2'] = df.text_clean_l1.apply(clean_l2)
```

```python id="a46TQFx8oDHU" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1597587420633, "user_tz": -330, "elapsed": 1860, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3db75b32-7824-4723-e775-5351d1b9e932"
df.drop_duplicates().replace(r'^\s*$', np.nan, regex=True).info()
```

```python id="LUqZormYoalz" colab={"base_uri": "https://localhost:8080/", "height": 385} executionInfo={"status": "ok", "timestamp": 1597590999123, "user_tz": -330, "elapsed": 3587, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="90ad9e43-c1ee-4f46-d495-45259adf9aea"
xx = df.text_clean_l2.apply(lambda x: len(x.split())).values
sns.distplot(xx[xx<1000])
xx.min(), xx.max()
```

```python id="4C0Y-gtbpnot"
# print(np.argmax(-xx))
print(list(np.argsort(-xx))[:20])
df.iloc[[374]]
```

```python id="2G7AVKDSw4oc" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1597589767548, "user_tz": -330, "elapsed": 1106, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d4b0f2da-85f0-4929-e768-d563aa839367"
df1 = df[(df.text_clean_l2.apply(lambda x: len(x.split()))>3) & (df.text_clean_l2.apply(lambda x: len(x.split()))<200)]
df1.info()
```

```python id="sk4hc_j0Ak4W"
def clean_l3(text):
  text = re.sub(r'[^a-z]', '', text)
  text = ' '.join([lemmatizer.lemmatize(w, 'v') for w in text.split()])
  text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
  text = ' '.join([w for w in text.split() if not w in stopwords])
  text = ' '.join([w for w in text.split() if not w in custom_stopwords])
  # seen = set()
  # seen_add = seen.add
  # text = ' '.join([x for x in text.split() if not (x in seen or seen_add(x))])
  text = ' '.join([ps.stem(w) for w in text.split()])
  return text
```

```python id="97G3Ro9MyLZs"
# def temp(text):
#   text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
#   text = ' '.join([lemmatizer.lemmatize(w, 'j') for w in text.split()])
#   text = ' '.join([lemmatizer.lemmatize(w, 'V') for w in text.split()])
#   text = ' '.join([lemmatizer.lemmatize(w, 'R') for w in text.split()])
#   return text

# # temp('communicate communication')

# import spacy
# from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
# lemmatizer = nlp.vocab.morphology.lemmatizer
# lemmatizer('communicate communication', VERB)

# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
# for w in ['commute', 'communication']:
#     rootWord=ps.stem(w)
#     print(rootWord)
```

```python id="uCbKgK3aBaFk"
df['text_clean_l2'] = df.text_clean_l1.apply(clean_l2)
```

```python id="h70Rr1RQExjJ"
df.text_clean_l2.sample(5, random_state=10).tolist()
```

```python id="lh5d0TOW076y"
# import spacy
# nlp = spacy.load("en_core_web_sm")

# def ners(text):
#   doc = nlp(text)
  # for token in doc:
  #   print(token.text)
  # x = list(set([ent.text for ent in doc.ents if ent.label_=='ORG']))
  # x = list(set([(ent.text,ent.label_) for ent in doc.ents]))
  # return x

# df.sample(20).text.apply(ners).tolist()
# df.text.iloc[[14]].apply(ners)
```

```python id="hbEjF9mzUhLM" colab={"base_uri": "https://localhost:8080/", "height": 51} executionInfo={"status": "ok", "timestamp": 1597523725451, "user_tz": -330, "elapsed": 3680, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e699411a-0511-4843-966b-7d5de2fd3ff4"
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_df=0.5, min_df=10, ngram_range=(1,3))
vectorizer.fit_transform(df.clean.tolist())
```

```python id="W1BKNJkcnR6y"
idx = 100
print(df.text.iloc[[idx]].tolist()[0])
pd.DataFrame(vectorizer.inverse_transform(vectorizer.transform([df.clean.tolist()[idx]]))).T[0]
```

```python id="KDsuAuhrnv5t"
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=10, ngram_range=(1,2))
X = vect.fit_transform(df.clean.tolist())

def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df
def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

features = vect.get_feature_names()
```

```python id="LTbFDHehr05B"
idx = 14
print(df.text.iloc[[idx]].tolist()[0])
print(top_feats_in_doc(X, features, idx, 10))
```

```python id="4TS9jDxjr78N"
df[df.clean.str.contains('vora')]
```

<!-- #region id="6Ogygp5uUl3M" -->
## Transformer model
<!-- #endregion -->

```python id="e75ADkO8pQ7m"
!pip install -q clean-text[gpl] && cp '/content/drive/My Drive/clean_v2.py' .
from clean_v2 import clean_l1
```

```python id="qGptuyeHpU6X"
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import csv

import warnings
warnings.filterwarnings("ignore")

tqdm.pandas()
%reload_ext autoreload
%autoreload 2
%reload_ext google.colab.data_table
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')
```

```python id="E7MFTAy4pXn_" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1598069413482, "user_tz": -330, "elapsed": 11692, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f6f6f973-5956-4fc8-ffb0-b6cf50f14998"
df_raw = pd.read_pickle(os.path.join(path,'data_clean_v2.p'))
df_raw.info()
```

```python id="CQnFjdJdppVQ"
df = df_raw.sample(10000, random_state=42)
```

```python id="gUPvgVrhpr18"
tokenlist = ['emailtoken', 'urltoken', 'newlinetoken', 'htmltoken', 'currencytoken', 'token', 'digittoken', 'numbertoken']
def preprocess(text):
  text = text.lower()
  text = ' '.join([w for w in text.split() if w not in tokenlist])
  text = ' '.join(text.split()[:50])
  return text
df['text'] = df.text.apply(preprocess)
df = df[df.text.apply(lambda x: len(x.split()))>3]
```

```python id="cJHwGqArqVs0"
df['target'] = df.target.apply(clean_l1).str.lower().str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' ')
minority_labels = df.target.value_counts()[df.target.value_counts()<100].index.tolist()
df['target'] = df.target.replace(dict.fromkeys(minority_labels, 'other'))
```

```python id="NpKHHoUwqhTu"
df = df[df.target!='other']
df.target.value_counts()[:25]
```

```python id="D1NARewJsTbD"
target_map = {'<redacted>': '<redacted>'}
df = df.replace({'target': target_map})
df.target.value_counts()[:25]
```

```python id="KwyuDewhypDp"
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])
```

```python id="gDNgxfml1c5P"
df.head()
```

```python id="kTtGC-_y0cGu"
with open('label.csv', 'w') as f:
    wr = csv.writer(f,delimiter="\n")
    wr.writerow(df.target.unique().tolist())
```

```python id="iEzYr_2mt2-K"
train, val = train_test_split(df, test_size=0.2, random_state=42)
train.reset_index(drop=True).to_csv('train.csv')
val.reset_index(drop=True).to_csv('val.csv')
```

```python id="AFVdzpaM15bF"
!pip install -q fast-bert
```

```python id="Bh7VEErg18sh" colab={"base_uri": "https://localhost:8080/", "height": 115, "referenced_widgets": ["dba40726134d460295836a4cb566a822", "684ddbad6fc64518b951d6c679d5f7c7", "439858897eef4ebd80cb4eca13aa2da9", "bcc9350609a248578e6d4c66a4160c09", "92f940f1d71d4d6d9b9a0a49b59af1f0", "d354c421b44842348af67c137364a543", "623adbc81043448393e131e430d4b1e8", "4c0bd4b8e0dd47ad85d95faefeb1df15", "17c93814cb01446fba6f1b3bc1e57f4c", "1f64da43af724514b4fbc5215f3fcbe4", "318d70924d4147a0b99dcd73595a48c2", "72297d4132fa4927ad35e4dec44b7ec2", "bcccbeea95db4985abf2512c8caf447a", "21fdf8c35de0432f96830a65ac761cc3", "f79a5a934626465987e6cdd559dbde46", "313d7079d5b74aee8d91e32a01408d7c"]} executionInfo={"status": "ok", "timestamp": 1597943546060, "user_tz": -330, "elapsed": 40641, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="63ba6376-1879-4570-d025-940107c38864"
from fast_bert.data_cls import BertDataBunch

databunch = BertDataBunch('/content', '/content',
                          tokenizer='distilbert-base-uncased',
                          train_file='train.csv',
                          val_file='val.csv',
                          label_file='label.csv',
                          text_col='text',
                          label_col='target',
                          batch_size_per_gpu=16,
                          max_seq_length=100,
                          multi_gpu=False,
                          multi_label=False,
                          model_type='distilbert')
```

```python id="EU_8JOp22G3j"
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
import torch

logger = logging.getLogger()
device_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
metrics = [{'name': 'accuracy', 'function': accuracy}]
```

```python id="iifY-pOp2YU3" colab={"base_uri": "https://localhost:8080/", "height": 171, "referenced_widgets": ["dd6c528596744531b479084c32a7a67d", "929f27b840a24526a806ede845ac1b97", "b6d92920a2434c2f8840e515ec5a7877", "631f2559878c40719df681d781b229ae", "4d3f5288eb0c46c69d0923a2b19714cf", "bc29c3bdf3e84abc919ec6bbc09403c5", "09acb85110ef4ac6b604225f4f403541", "56354758183f463a9dffa4590149c6f2"]} executionInfo={"status": "ok", "timestamp": 1597943583584, "user_tz": -330, "elapsed": 77856, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d1a59933-3bf1-49b9-8ed6-dfc5ef0c9572"
learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='distilbert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir='/content',
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=False,
						is_fp16=True,
						multi_label=False,
						logging_steps=50)
```

```python id="ZbQQhEJ62dY2" colab={"base_uri": "https://localhost:8080/", "height": 450, "referenced_widgets": ["ab7e1e874d8e4173b50b54ae4a6f8f9e", "763432504f734cfeaf0bef1cd68e72e9", "4265c59eafcd4d8f962960cf8cc2ac0d", "1fa5de508fae4cc6be576f118d8440a7", "bac660ab018b42799d198ad93b94d0f0", "0841021295ab4c6592328b198d011ceb", "cf33508e57d740a4b0eca9b9bbecee22", "958428cf14ab451fb3d111655e305154"]} executionInfo={"status": "ok", "timestamp": 1597943955981, "user_tz": -330, "elapsed": 450154, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bb664b40-44ca-4e8b-991d-6da9f4821061"
learner.lr_find(start_lr=1e-4,optimizer_type='lamb')
```

```python id="v63xoVnp2hjx" colab={"base_uri": "https://localhost:8080/", "height": 401} executionInfo={"status": "ok", "timestamp": 1597944129491, "user_tz": -330, "elapsed": 2545, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="33dd6a6c-428f-4161-baa3-6a31f8c739f5"
learner.plot(show_lr=2e-2)
```

```python id="k-O9OQVT3Lk8" colab={"base_uri": "https://localhost:8080/", "height": 168} executionInfo={"status": "ok", "timestamp": 1597944555033, "user_tz": -330, "elapsed": 88988, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="066b343a-954e-4f9e-ae74-e38b0566a1c4"
 learner.fit(epochs=1,
            lr=2e-2,
            validate=True,
            schedule_type="warmup_cosine",
            optimizer_type="lamb",
            return_results=True)
```

```python id="ujFx3Fxx3LiO" colab={"base_uri": "https://localhost:8080/", "height": 54} executionInfo={"status": "ok", "timestamp": 1597944461164, "user_tz": -330, "elapsed": 5574, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="88f57bfd-e0c8-4011-eed8-6131667b0d32"
learner.validate()
```

```python id="FEAcnoDn3Ld9"
learner.save_model()
```

```python id="-0VZJzq47Xzh"
xx = val.sample(5); xx
```

```python id="ID9Fvoqj71RN" colab={"base_uri": "https://localhost:8080/", "height": 505} executionInfo={"status": "ok", "timestamp": 1597944971417, "user_tz": -330, "elapsed": 1626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1e41fe26-2b25-418c-88e0-0db2241a6cef"
predictions = learner.predict_batch(xx.text.tolist())
pd.DataFrame(predictions).T
```

```python id="GkZWwZfZ3abu"
# from fast_bert.prediction import BertClassificationPredictor
# MODEL_PATH = '/content/model_out'

# predictor = BertClassificationPredictor(
# 				model_path='/content',
# 				label_path='/content',
# 				multi_label=False,
# 				model_type='xlnet',
# 				do_lower_case=True)

# single_prediction = predictor.predict("just get me result for this text")
# texts = ["this is the first text", "this is the second text"]
# multiple_predictions = predictor.predict_batch(texts)
```

<!-- #region id="8-k7MSeU5N9O" -->
---
<!-- #endregion -->

```python id="kHEcJPkL5Okh"
train, val = train_test_split(df, test_size=0.2, random_state=42)

train = train.reset_index(drop=True)
train.columns = ['text','labels']

val = val.reset_index(drop=True)
val.columns = ['text','labels']
```

```python id="sxJsGaPf54qW"
!pip install -q simpletransformers
```

```python id="bxcPfHyZ6Cuq"
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
```

```python id="QPpKZFw96Cs6"
model_args = ClassificationArgs(num_train_epochs=1, learning_rate=1e-2)
```

```python id="qJ9TyF-v6CoZ"
model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=df.target.nunique(), args=model_args)
```

```python id="7w7ZKHRI6Cm6"
model.train_model(train)
```

```python id="poKWYOpG6Clm"
scores1, model_outputs, wrong_predictions = model.eval_model(val)
```

```python id="6vCg2i2P6Ch6"
scores1
```

```python id="vCkX4oSU6CgS"
from sklearn.metrics import f1_score, accuracy_score
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')
```

```python id="R866_BHt65TT"
scores2, model_outputs, wrong_predictions = model.eval_model(val, f1=f1_multiclass, acc=accuracy_score)
```

```python id="RgCPQKny3wbx"
scores2
```

```python id="bHg1kdJJ7DUC"
predictions, raw_output  = model.predict(['<redacted>'])
```

<!-- #region id="edV2IZYFXwXk" -->
## TFIDF model
<!-- #endregion -->

```python id="w2rxTQFVwUs6"
!pip install -q clean-text[gpl] && cp '/content/drive/My Drive/clean_v2.py' .

from clean_v2 import clean_l1
```

```python id="BCxKJQN9D0HJ"
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = list(set(stopwords.words('english')))

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') 
lemmatizer = WordNetLemmatizer()

from nltk.stem import PorterStemmer
ps = PorterStemmer()

from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore")

tqdm.pandas()
%reload_ext autoreload
%autoreload 2
%reload_ext google.colab.data_table
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')
```

```python id="dHaDj6kED5dN" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1598025748756, "user_tz": -330, "elapsed": 12885, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9e0505ce-0f74-4989-9a0b-195a5eb9fd7d"
df_raw = pd.read_pickle(os.path.join(path,'data_clean_v2.p'))
df_raw.info()
```

```python id="P1trnRboFlKa"
df = df_raw.sample(10000, random_state=42)
```

```python id="K_dD9JNuEYqI"
def clean_l2(text):
  text = text.lower()
  text = re.sub(r'[^a-z ]', '', text)
  text = ' '.join([lemmatizer.lemmatize(w, 'v') for w in text.split()])
  text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
  text = ' '.join([w for w in text.split() if not w in stopwords])
  text = ' '.join([ps.stem(w) for w in text.split()])
  return text
```

```python id="OpcrR8EYE95N"
from collections import OrderedDict
df['target'] = df.target.apply(clean_l1).str.lower().str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' ')
```

```python id="BQFHM1LnGedJ"
df['text'] = df['text'].apply(clean_l2)
```

```python id="XUVjS-bYzTzX"
df = df[df.text.apply(lambda x: len(x.split()))>3]
```

```python id="WNi46YPuFpcZ"
xx = df.sample(5, random_state=40)
xx
```

```python id="VMysCgg_8sg2"
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
```

```python id="xPDY-ViLxMWI" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1598026015093, "user_tz": -330, "elapsed": 1749, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3c0eeb62-e9a5-4c47-c95d-59918dea030b"
df['text'].apply(lambda x: len(x.split(' '))).sum()
```

```python id="YukuB5cqypYr"
minority_labels = df.target.value_counts()[df.target.value_counts()<100].index.tolist()
df['target'] = df.target.replace(dict.fromkeys(minority_labels, 'Other'))
df = df[df.target!='Other']
```

```python id="gpe9jMv0xdnn"
X = df.text
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
```

```python id="3MLtHV2ExvSd" colab={"base_uri": "https://localhost:8080/", "height": 306} executionInfo={"status": "ok", "timestamp": 1598026355049, "user_tz": -330, "elapsed": 2131, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a04a8225-c64f-4af9-d632-a3b1dc79bae3"
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)
```

```python id="vOK-cf2gyDCa"
label_list = list(df.target.unique())
```

```python id="qcTbWA9Bx0pY"
%%time
from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=label_list))
```

```python id="OT2kXPSLx5RD" colab={"base_uri": "https://localhost:8080/", "height": 374} executionInfo={"status": "ok", "timestamp": 1598026475330, "user_tz": -330, "elapsed": 2058, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8138ec8a-5486-43db-e50f-b5be04373221"
from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)
```

```python id="vqbX9-eRydlf"
%%time
y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=label_list))
```

```python id="vR1em2v7zXvm" colab={"base_uri": "https://localhost:8080/", "height": 391} executionInfo={"status": "ok", "timestamp": 1598026559596, "user_tz": -330, "elapsed": 8725, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1036625a-4632-4b94-9a74-048782a1c8e2"
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)
```

```python id="XOs8bkx3zg24"
%%time
y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=label_list))
```

```python id="Dt1107X2zkyj"
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import re
```

```python id="3CWldrzcz4vf"
def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the post.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled
```

```python id="W-U6Th0yz8o3"
X_train, X_test, y_train, y_test = train_test_split(df.text, df.target, random_state=0, test_size=0.3)
X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
all_data = X_train + X_test
```

```python id="9zac3Ygi0DMc" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1598026783023, "user_tz": -330, "elapsed": 6872, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="573703f7-3589-4017-9d65-d6a0fd89f2ed"
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(all_data)])
```

```python id="xg0Fbuyf0X2v"
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
```

```python id="X3ENGKxc0caJ"
def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors
```

```python id="hwAVgSAU0g0k"
train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')
```

```python id="XdX8xFBb0kyY" colab={"base_uri": "https://localhost:8080/", "height": 102} executionInfo={"status": "ok", "timestamp": 1598026839795, "user_tz": -330, "elapsed": 3244, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bd6e6da0-a839-4cd8-9dfe-f156008ca53c"
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(train_vectors_dbow, y_train)
```

```python id="Edtpau890kwg"
y_pred = logreg.predict(test_vectors_dbow)
```

```python id="V5n-mojn0krx"
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=label_list))
```

```python id="oruSKku20wfa"
import itertools
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
```

```python id="iNb9hzDI082w" colab={"base_uri": "https://localhost:8080/", "height": 51} executionInfo={"status": "ok", "timestamp": 1598026935794, "user_tz": -330, "elapsed": 1487, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b279f9be-656d-454a-aef6-95e21cc92542"
train_size = int(len(df) * .7)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(df) - train_size))
```

```python id="Du9CVz-J0-eK"
train_posts = df['text'][:train_size]
train_tags = df['target'][:train_size]

test_posts = df['text'][train_size:]
test_tags = df['target'][train_size:]
```

```python id="G_HFk3_Q1AiU"
max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
```

```python id="V2zpZfyo1H0N"
tokenize.fit_on_texts(train_posts) # only fit on train
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)
```

```python id="Syys4hmy1Je8"
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)
```

```python id="j7SRaWY01LPo"
num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
```

```python id="uJI2IDyk1Vf5" colab={"base_uri": "https://localhost:8080/", "height": 85} executionInfo={"status": "ok", "timestamp": 1598027039642, "user_tz": -330, "elapsed": 1878, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cb9df911-b3c2-433d-9393-016aa556b95e"
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
```

```python id="qWIAD7V41Xut"
batch_size = 32
epochs = 2
```

```python id="LYauGei31ZP9"
# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

```python id="4ck3MgmI1axp" colab={"base_uri": "https://localhost:8080/", "height": 85} executionInfo={"status": "ok", "timestamp": 1598027064606, "user_tz": -330, "elapsed": 3661, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eacabdaa-3eac-4bd4-ac88-4ea2ba142e0e"
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
```

```python id="sbqcsfWS1dZ9" colab={"base_uri": "https://localhost:8080/", "height": 51} executionInfo={"status": "ok", "timestamp": 1598027074869, "user_tz": -330, "elapsed": 2953, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="72dcd089-e432-408b-a9d8-f19e9f6b30ad"
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])
```

```python id="-8dqS1akb9lD" colab={"base_uri": "https://localhost:8080/", "height": 51} executionInfo={"status": "ok", "timestamp": 1598104750040, "user_tz": -330, "elapsed": 2505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d9ef6a95-3c0d-4420-a7d6-317c770e61a5"
def checkinside(V, T):
  lAB = ((V[1][0] - V[0][0])**2 + (V[1][1] - V[0][1])**2)**(0.5)
  lBC = ((V[2][0] - V[1][0])**2 + (V[2][1] - V[1][1])**2)**(0.5)
  uAB = ((V[1][0] - V[0][0]) / lAB, (V[1][1] - V[0][1]) / lAB)
  uBC = ((V[2][0] - V[1][0]) / lBC, (V[2][1] - V[1][1]) / lBC)
  BP = ((T[0][0] - V[1][0]), (T[0][1] - V[1][1]))
  SignedDistABP = BP[0] * uAB[1] - BP[1] * uAB[0]
  SignedDistBCP = - BP[0] * uBC[1] + BP[1] * uBC[0]
  result = 'inside' if ((SignedDistABP*SignedDistBCP > 0) and \
                        (abs(SignedDistABP) <= lBC) and \
                        abs(SignedDistBCP) <= lAB) \
                        else 'not inside'
  return result

V = [(670273, 4879507), (677241, 4859302), (670388, 4856938), (663420, 4877144)]
T = [(670831, 4867989), (675097, 4869543)]
print(checkinside(V,[T[0]]))
print(checkinside(V,[T[1]]))
```

<!-- #region id="twJPvJEGXyKk" -->
## FastText model
<!-- #endregion -->

```python id="Y5VvWh-WaTne"
!pip install -q clean-text[gpl] && cp '/content/drive/My Drive/clean_v2.py' .

from clean_v2 import clean_l1
```

```python id="b77ZKHQCRzuJ"
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

tqdm.pandas()
%reload_ext autoreload
%autoreload 2
%reload_ext google.colab.data_table
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')
```

```python id="k37X92FzSEwz" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1597917161960, "user_tz": -330, "elapsed": 11302, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="55ed77cd-1275-46ea-e44f-a8cc98a73512"
df_raw = pd.read_pickle(os.path.join(path,'data_clean_v2.p'))
df_raw.info()
```

```python id="mxCQwouYZ-Ql"
df = df_raw.sample(100000, random_state=42)
```

```python id="h7DefE4RSL0Y"
# !pip install fasttext
```

```python id="h0Kgbtg5jFBu"
# preprocessing
# lowercase
# remove tokens
# truncate post-regards
# lower th remove
# upper th truncate
# clean categories
# collate categories
# train test split

# tokenlist = ' '.join([i for i in df['text']]).split()
# tokenlist = list(set([w for w in tokenlist if 'token' in w]))
tokenlist = ['emailtoken', 'urltoken', 'htmltoken', 'currencytoken', 'token', 'digittoken', 'numbertoken']

def preprocess(text):
  text = text.lower()
  text = ' '.join([w for w in text.split() if w not in tokenlist])
  text = ' '.join(text.split()[:50])
  return text
```

```python id="GJcUoM_ez8UH" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1597926142587, "user_tz": -330, "elapsed": 1068, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="57d05561-5682-459c-a894-00f86b56f5a8"
print(tokenlist)
```

```python id="J-69986Ymk_d"
# xx = df.sample()
# xx
df['text'] = df.text.apply(preprocess)
df = df[df.text.apply(lambda x: len(x.split()))>3]
# preprocess(xx.text.tolist()[0])
```

```python id="UHd3lGOqak4-"
# from collections import OrderedDict
df['target'] = df.target.apply(clean_l1).str.lower().str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' ')
```

```python id="Iqd33UCirrzx"
df.sample(5)
```

```python id="oGOlYA3GtyjX"
df.target.value_counts()[:20]
```

```python id="DV1k-Uazsmet"
# sns.distplot(df.target.value_counts())
```

```python id="l7TRmf6Ot_fR"
minority_labels = df.target.value_counts()[df.target.value_counts()<100].index.tolist()
df['target'] = df.target.replace(dict.fromkeys(minority_labels, 'Other'))
```

```python id="M1Wb7nf5uFKx" colab={"base_uri": "https://localhost:8080/", "height": 382} executionInfo={"status": "ok", "timestamp": 1597926235447, "user_tz": -330, "elapsed": 3572, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7a040b88-eb71-4b67-f3ac-d33ebbb29db2"
df = df[df.target!='Other']
sns.distplot(df.target.value_counts());
```

```python id="6xoPSW7qaqyV"
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df.target)
```

```python id="qqccvFYydi2g" colab={"base_uri": "https://localhost:8080/", "height": 68} executionInfo={"status": "ok", "timestamp": 1597926248196, "user_tz": -330, "elapsed": 824, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8f41991d-7542-491d-e11c-b3d188876e03"
df.isna().any()
```

```python id="quc1nKu1dlcG"
df['target'] = ['__label__'+str(s) for s in df['target']]
df = df[['target','text']]
```

```python id="d04SXHjauhaK"
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
```

```python id="6VlaKqzTfuI5"
import csv
train.to_csv('train.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
test.to_csv('test.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
```

```python id="CMAB78pAfS8v"
# train.sample(5).to_csv('sample.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
```

```python id="f0yzMHb5gZIp"
# import fasttext
# model = fasttext.train_supervised(input='train.txt', epoch=50) --> (17874, 0.5248405505203089, 0.5248405505203089)
# model = fasttext.train_supervised(input='train.txt', epoch=50, lr=0.5, wordNgrams=2, loss='hs') --> (17874, 0.46620789974264293, 0.46620789974264293)
# model = fasttext.train_supervised(input='train.txt', --> (17874, 0.4858453619782925, 0.4858453619782925)
#                                   epoch=25,
#                                   lr=0.2,
#                                   loss='hs',
#                                   autotuneMetric='f1',
#                                   verbose=5,
#                                   minCount=10,
#                                   )
# model = fasttext.train_supervised(input='train.txt', --> (17874, 0.5262392301667226, 0.5262392301667226)
#                                   epoch=50,
#                                   lr=0.1,
#                                   loss='softmax',
#                                   autotuneMetric='f1',
#                                   verbose=5,
#                                   minCount=20,
#                                   )
model = fasttext.train_supervised(input='train.txt', 
                                  epoch=50,
                                  lr=0.1,
                                  loss='softmax',
                                  autotuneMetric='f1',
                                  verbose=5,
                                  minCount=20,
                                  )
```

```python id="n9EgtOfghzi5" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1597927998097, "user_tz": -330, "elapsed": 215881, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="47f32006-ac94-4dd6-eed5-fe6697257057"
model.test("test.txt", k=1)
```

```python id="EqnP3RyAygzs" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1597926832763, "user_tz": -330, "elapsed": 2453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0e024b71-cb25-4b70-fea6-392fb6508839"
model.test("test.txt", k=5)
```

```python id="t7juI-eVhRjo" colab={"base_uri": "https://localhost:8080/", "height": 110} executionInfo={"status": "ok", "timestamp": 1597927022572, "user_tz": -330, "elapsed": 1139, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0255e384-e78c-412e-9b4c-cb15ea786979"
xx = df.sample(); xx
```

```python id="CJFvEgHIhcG3" colab={"base_uri": "https://localhost:8080/", "height": 68} executionInfo={"status": "ok", "timestamp": 1597927022574, "user_tz": -330, "elapsed": 891, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="06383cff-b69a-4306-a3a0-2c6d493bd09d"
model.predict(xx.text.tolist()[0], k=5)
```

<!-- #region id="hHXMC73lYsr4" -->
## Pipeline
<!-- #endregion -->

```python id="KiF2mUb9fAqS"
!pip install -q fast-bert
!pip install -q fasttext
!pip install -q clean-text[gpl]
```

```python id="8kJPedUNcCAq"
# setup
import os
import pickle
import shutil
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings("ignore")
Path(work_path).mkdir(parents=True, exist_ok=True)
os.chdir(work_path)
```

```python id="15As3RASYIYP"
shutil.copyfile(os.path.join(save_path,'utils_clean.py'), os.path.join(work_path,'utils_clean.py'))
from utils_clean import clean_l1, clean_l2

shutil.copyfile(os.path.join(save_path,'utils_preprocess.py'), os.path.join(work_path,'utils_preprocess.py'))
from utils_preprocess import *

shutil.copyfile(os.path.join(save_path,'label_encoder.p'), os.path.join(work_path,'label_encoder.p'))
label_encoder = pickle.load(open('label_encoder.p', 'rb'))

shutil.copyfile(os.path.join(save_path,'label_map.p'), os.path.join(work_path,'label_map.p'))
label_map = pickle.load(open('label_map.p', 'rb'))

import fasttext
shutil.copyfile(os.path.join(save_path,'fasttext.bin'), os.path.join(work_path,'fasttext.bin'))
model_fasttext = fasttext.load_model('fasttext.bin')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
vectorizer = CountVectorizer(ngram_range=(1,1))
shutil.copyfile(os.path.join(save_path,'model_countvectorizer_large.p'), os.path.join(work_path,'model_countvectorizer.p'))
model_countvectorizer = pickle.load(open('model_countvectorizer.p', 'rb'))
dtmx = model_countvectorizer['dtm']
vectorizerx = model_countvectorizer['vectorizer']
target_categories = model_countvectorizer['target_categories']
target_labels = model_countvectorizer['target_labels']

shutil.copyfile(os.path.join(save_path,'model_tfidf_large.p'), os.path.join(work_path,'model_tfidf.p'))
model_tfidf = pickle.load(open(os.path.join(work_path,'model_tfidf.p'), 'rb'))

from fast_bert.prediction import BertClassificationPredictor
shutil.unpack_archive(os.path.join(save_path,'fastbert_large_iter2.zip'), work_path, 'zip')
MODEL_PATH = os.path.join(work_path, 'model_out')
model_fastbert = BertClassificationPredictor(
                    model_path=MODEL_PATH,
                    label_path=work_path,
                    multi_label=False,
                    model_type='distilbert',
                    do_lower_case=True)
```

```python id="jyekm-vdHzSo"
X = pd.DataFrame(label_map, index=[0]).T.reset_index()
X.columns = ['Raw_data_labels','processed_labels']
X.to_csv('')
```

```python id="UCqjgBOscTsn" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1598870887239, "user_tz": -330, "elapsed": 41252, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fcc50ed7-60ef-450f-bd13-13ea1935cf72"
df_raw = pd.read_pickle(os.path.join('/content/drive/My Drive/','data','raw','data_raw.p'))
df_raw.info()
```

```python id="ISpHU-TQIQhx"
df_raw['labels_raw'] = df_raw['QueryType'] + ' | ' + df_raw['SubQueryType']
df_raw['labels_raw'].value_counts().to_csv('labels_rawdata.csv')
```

```python id="KuiKRsQaXm8F"
# test_set = df_raw.sample(50000, random_state=10)
# def preprocess(X):
#   X = X.drop_duplicates()
#   X = X.dropna(subset=['tSubject', 'mMsgContent'], how='all')
#   X['type'] = X['QueryType'] + ' | ' + X['SubQueryType']
#   X['type'] = X['type'].fillna(' ')
#   X['type_orig'] = X['type']
#   X['type'] = X['type'].apply(clean_l1).str.lower().str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' ').apply(clean_l1)
#   X = X[X['type'].isin(list(label_encoder.classes_))]
#   X['subj&msg'] = X['tSubject'] + ' sub_eos_token ' + X['mMsgContent']
#   X = X[['subj&msg','type', 'type_orig']]
#   X.columns = ['text','target', 'type_orig']
#   X = X.dropna()
#   return X
# test_set = preprocess(test_set)
# test_set.describe()

# label_map = test_set[['type_orig','target']].set_index('type_orig').to_dict()['target']
# pickle.dump(label_map, open(os.path.join(#save_path, 'label_map.p'), 'wb'))
```

```python id="pAZKv5cndsPA"
# test_set = df_raw.sample(10000, random_state=10)
# def preprocess(X):
#   X = X.drop_duplicates()
#   X = X.dropna(subset=['tSubject', 'mMsgContent'], how='all')
#   X['type'] = X['QueryType'] + ' | ' + X['SubQueryType']
#   X['type'] = X['type'].fillna(' ')
#   X['type'] = X['type'].apply(clean_l1).str.lower().str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' ').apply(clean_l1)
#   X = X[X['type'].isin(list(label_encoder.classes_))]
#   X['subj&msg'] = X['tSubject'] + ' sub_eos_token ' + X['mMsgContent']
#   X = X[['subj&msg','type']]
#   X.columns = ['text','target']
#   X = X.dropna()
#   return X
# test_set = preprocess(test_set)
# test_set.describe()
```

```python id="gFI4fLzNeXrv"
  ##### <----- freezed backup -----> #####

# def predict_fasttext(text):
#   text = clean_l1(text)
#   text = preprocess_fasttext(text)
#   preds = model_fasttext.predict(text, k=-1)
#   label_names = label_encoder.inverse_transform([int(x.split('__')[-1]) for x in preds[0]])
#   preds = [(x,y) for x,y in zip(label_names,preds[1])]
#   return preds

# def predict_fastbert(text):
#   text = clean_l1(text)
#   text = preprocess_fastbert(text)
#   preds = model_fastbert.predict(text)
#   preds = [(label_encoder.inverse_transform([int(x[0])])[0],x[1]) for x in preds]
#   return preds

# def predict_countvect(text):
#   text = clean_l1(text)
#   text = clean_l2(text)
#   text = preprocess_countvectorizer(text)
#   cosim = linear_kernel(vectorizerx.transform([text]), dtmx).flatten()
#   preds = [(target_categories[i],cosim[i]) for i in range(len(cosim))]
#   preds = [target_categories[x] for x in np.argsort(-cosim)[:20]]
#   return preds

# def predict_tfidf(text):
#   text = clean_l1(text)
#   text = clean_l2(text)
#   preds = model_tfidf.predict_proba([text])[0]
#   preds = [(label_encoder.inverse_transform([int(x)])[0],y) for x,y in zip(model_tfidf.classes_, preds)]
#   preds.sort(key = lambda x: x[1], reverse=True)  
#   return preds


# query = test_set.sample()
# print('Text: ',query.text.values[0])
# print('Actual Label: ',query.target.values[0])
# print('Predicted Labels: ')
# pred1 = predict_fasttext(query.text.values[0])
# pred2 = predict_fastbert(query.text.values[0])
# pred3 = predict_countvect(query.text.values[0])
# pred4 = predict_tfidf(query.text.values[0])
```

```python id="qj8t9ovJfkK8"
# lmr = {label_map[v]:v for v in label_map.keys()}
```

```python id="bXERZM74br8l"
def predict_fasttext(text):
  text = clean_l1(text)
  text = preprocess_fasttext(text)
  preds = model_fasttext.predict(text, k=-1)
  preds = [int(x.split('__')[-1]) for x in preds[0]]
  preds = pd.DataFrame([(x,(1/(i+1))) for i,x in enumerate(preds)],
                       columns=['label','rank_fasttext']).set_index('label')
  return preds

def predict_fastbert(text):
  text = clean_l1(text)
  text = preprocess_fastbert(text)
  preds = model_fastbert.predict(text)
  preds = pd.DataFrame([(int(x[0]),(1/(i+1))) for i,x in enumerate(preds)],
                      columns=['label','rank_fastbert']).set_index('label')
  return preds

def predict_tfidf(text):
  text = clean_l1(text)
  text = clean_l2(text)
  preds = model_tfidf.predict_proba([text])[0]
  preds = [(int(x),y) for x,y in zip(model_tfidf.classes_, preds)]
  preds.sort(key = lambda x: x[1], reverse=True)
  preds = pd.DataFrame([(int(x[0]),(1/(i+1))) for i,x in enumerate(preds)],
                    columns=['label','rank_tfidf']).set_index('label')
  return preds

def predict_countvect(text):
  text = clean_l1(text)
  text = clean_l2(text)
  text = preprocess_countvectorizer(text)
  cosim = linear_kernel(vectorizerx.transform([text]), dtmx).flatten()
  preds = [(int(target_categories[i]),cosim[i]) for i in range(len(cosim))]
  preds.sort(key = lambda x: x[1], reverse=True)
  preds = pd.DataFrame([(int(x[0]),(1/(i+1))) for i,x in enumerate(preds)],
                  columns=['label','rank_cvt']).set_index('label')
  return preds

model_weight = {'fasttext':10, 'fastbert':5, 'tfidf':3, 'cvt':2}

def predict(text):
  pred = predict_fasttext(text)
  pred = pred.join(predict_fastbert(text), on='label')
  pred = pred.join(predict_tfidf(text), on='label')
  pred = pred.join(predict_countvect(text), on='label')
  pred['score'] = (pred['rank_fasttext']*model_weight['fasttext']) + \
  (pred['rank_fastbert']*model_weight['fastbert']) + \
  (pred['rank_tfidf']*model_weight['tfidf']) + \
  (pred['rank_cvt']*model_weight['cvt'])
  pred = pred.sort_values(by='score', ascending=False)
  return pred

def predict(text):
  pred = predict_fasttext(text)
  pred = pred.join(predict_tfidf(text), on='label')
  pred = pred.join(predict_countvect(text), on='label')
  pred['score'] = (pred['rank_fasttext']*model_weight['fasttext']) + \
  (pred['rank_tfidf']*model_weight['tfidf']) + \
  (pred['rank_cvt']*model_weight['cvt'])
  pred = pred.sort_values(by='score', ascending=False)
  return pred
```

```python id="I0x4viGLdUp0"
testx = pd.read_excel(os.path.join(save_path,'TSD_v1.xlsx'), sheet_name='Database records')
testx.head()
```

```python id="kunTKw5Df5W0"
print(testx.info())
X = testx[(testx['OriginalQuery']+ ' | '+testx['OriginalSubQuery']).isin(list(set(label_map.keys())))]
print(X.info())
X.head()
```

```python id="1MuEmxv5fi3Q"
X['text'] = X['tSubject'] + ' sub_eos_token ' + X['mMsgContent']
X['label'] = X['OriginalQuery'] + ' | '+ X['OriginalSubQuery']
X['plabel'] = X['label'].apply(lambda x: label_map[x])
X['clabel'] = label_encoder.transform(X.plabel)
X = X[['text','clabel']]
print(X.info())
X = X.dropna().drop_duplicates()
print(X.info())
X.head()
```

```python id="wEtRLtiYirzF" colab={"base_uri": "https://localhost:8080/", "height": 51} executionInfo={"status": "ok", "timestamp": 1598812687665, "user_tz": -330, "elapsed": 1808921, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="335db9db-1786-44db-dbd0-89c825ac5c23"
from tqdm import tqdm
tqdm.pandas()
top1 = top2 = top3 = 0
for index, row in tqdm(X.iterrows(), total=X.shape[0]):
  text = row.text
  label = row.clabel
  preds = predict(text).index.tolist()[:3]
  if label==preds[0]:
    top1+=1
  elif label==preds[1]:
    top2+=1
  elif label==preds[2]:
    top3+=1
print(top1, top2, top3)
```

```python id="muSU6h9Hqc3l" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1598813603171, "user_tz": -330, "elapsed": 1007, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c44255dd-0d5d-4af0-b17d-15bc6dd2694d"
top1p = top1/X.shape[0]
top2p = top1p + top2/X.shape[0]
top3p = top2p + top3/X.shape[0]

print(top1p, top2p, top3p)
```

```python id="ay-dfCenbe8v" colab={"base_uri": "https://localhost:8080/", "height": 269} executionInfo={"status": "ok", "timestamp": 1598532374357, "user_tz": -330, "elapsed": 5335, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3d266c76-686e-47a5-f666-8c148aebc265"
query = test_set.sample()
# print('Text: ',query.text.values[0])
print('Actual Label: ',query.target.values[0])
print('Actual Label: ',label_encoder.transform([query.target.values[0]]))
predict(query.text.values[0]).head()
```

```python id="gajtJsfoyXVS"
test_set['target_cat'] = label_encoder.transform(test_set.target)
test_set.head()
```

```python id="x_cU7vl8vDCe" colab={"base_uri": "https://localhost:8080/", "height": 51} executionInfo={"status": "ok", "timestamp": 1598389586728, "user_tz": -330, "elapsed": 7238727, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b791ccf-c7b4-40dc-a163-6a75137d4afc"
from tqdm import tqdm
tqdm.pandas()
top1 = top2 = top3 = 0
for index, row in tqdm(test_set.iterrows(), total=test_set.shape[0]):
  text = row.text
  label = row.target_cat
  preds = predict(text).index.tolist()[:3]
  if label==preds[0]:
    top1+=1
  elif label==preds[1]:
    top2+=1
  elif label==preds[2]:
    top3+=1
print(top1, top2, top3)
```

```python id="QeHWkdVoz_ws" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1598389942491, "user_tz": -330, "elapsed": 1430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0dcdae98-f886-4712-fb93-d74ce0da6fe9"
top1p = top1/test_set.shape[0]
top2p = top1p + top2/test_set.shape[0]
top3p = top2p + top3/test_set.shape[0]

print(top1p, top2p, top3p)
```

```python id="leriRNFRdvIL"
def func(subj, msg):
  text = str(subj) + ' sub_eos_token ' + str(msg)
  preds = predict(text).head(3)
  preds.index = label_encoder.inverse_transform(preds.index.tolist())
  preds = preds.rename_axis('label').reset_index()[['label','score']]
  preds.label = preds.label.apply(lambda x: label_map[x])
  preds = preds.T.to_dict()
  preds = str(preds)
  return preds
```

```python id="-1gtjKrdYaUU"
x = test_set.sample()
x
```

```python id="vl3T4yCKYmvJ"
x.text.tolist()[0]
```

```python id="WPfidXIiYyTC"
subj = '<redacted>'
msg = '<redacted>'
```

```python id="Jm77CpiCC8QA"
xx = func(subj, msg)
xx
```

```python id="z_RKqaTOZFH2"
xx = func(subj, msg)
xx
```

```python id="G_66smUxZyH_"
pd.Series(test_set['type_orig'].unique()).to_csv('label_list.csv', index=False)
```

```python id="8DWczOcybExT"
!pip install -q gradio
import gradio as gr
```

```python id="ZL9uUzJ3cB68"
gr.Interface(func, 
             [
              gr.inputs.Textbox(lines=2, label='Email Subject'),
              gr.inputs.Textbox(lines=10, label='Email Body'),
             ],
             gr.outputs.Textbox(label='Output Labels')).launch();
```

```python id="xKIgGftUdIs2"
xx = pd.DataFrame(label_map, index=[0]).T.reset_index()
xx.columns = ['plabel', 'olabel']
xx.head()
```

```python id="tzqykAQ8Q6EI"
xx.olabel.value_counts()
```
