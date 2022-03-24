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

```python colab={"base_uri": "https://localhost:8080/"} id="tF-n7DX1QVs4" executionInfo={"status": "ok", "timestamp": 1622310894832, "user_tz": -330, "elapsed": 1522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="447b2005-abf3-4648-8489-5413babbc54f"
!wget -O deskdrop.zip https://github.com/sparsh-ai/reco-data/blob/master/DeskDrop.zip?raw=true
!unzip deskdrop.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="ri8DAP-SQn4W" executionInfo={"status": "ok", "timestamp": 1622312694346, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cbe68800-35ac-4adc-87cb-8aa66c271d4d"
import math
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
```

```python colab={"base_uri": "https://localhost:8080/"} id="71nZVfMgRBO1" executionInfo={"status": "ok", "timestamp": 1622310950621, "user_tz": -330, "elapsed": 451, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ce552b9f-8728-4d15-c014-4c52808dd6bc"
article_df = pd.read_csv('shared_articles.csv')
article_df.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 632} id="5UJuYL5vRGzS" executionInfo={"status": "ok", "timestamp": 1622310960048, "user_tz": -330, "elapsed": 495, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="61d27888-78ba-4d5a-d870-408c338676c1"
article_df.sample(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="cig7sygsRLR_" executionInfo={"status": "ok", "timestamp": 1622310985710, "user_tz": -330, "elapsed": 401, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4d2280d8-5b7c-4425-a1ba-a544b08803ae"
interactions_df = pd.read_csv('users_interactions.csv')
interactions_df.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="2PcuDqomRRpw" executionInfo={"status": "ok", "timestamp": 1622310990050, "user_tz": -330, "elapsed": 409, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0657f07e-6d5f-4b10-c688-0fc153acd68a"
interactions_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="2ahfi38-SKlc" executionInfo={"status": "ok", "timestamp": 1622311371734, "user_tz": -330, "elapsed": 417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="acff7c35-4d8d-49bc-9c1b-88e7708afa94"
# users can interact with aticles multiple times
interactions_df.groupby(['personId', 'contentId']).size()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 487} id="2vaXgIJzSKiz" executionInfo={"status": "ok", "timestamp": 1622311485637, "user_tz": -330, "elapsed": 680, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aca8b922-5b80-4220-a642-109fd169b850"
interactions_df[(interactions_df['personId']==9210530975708218054)].sort_values(by=['timestamp'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="xTiaBCX4SKgR" executionInfo={"status": "ok", "timestamp": 1622311696729, "user_tz": -330, "elapsed": 1598, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d6f46a12-1e25-441e-b033-ebccc5a1f2e8"
interactions_df[(interactions_df['personId']==-9223121837663643404) & (interactions_df['contentId']==-7423191370472335463)].sort_values(by=['timestamp'])
```

```python colab={"base_uri": "https://localhost:8080/"} id="LQadUlopSKdn" executionInfo={"status": "ok", "timestamp": 1622311834133, "user_tz": -330, "elapsed": 410, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bbca61b7-bed0-43b0-a905-7194b12c6005"
users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]

print('# total users: {}'.format(interactions_df['personId'].nunique()))
print('# users with at least 5 interactions: {}'.format(len(users_with_enough_interactions_df)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="TAWWdPzeSKa1" executionInfo={"status": "ok", "timestamp": 1622311891292, "user_tz": -330, "elapsed": 439, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="19d897f4-f5e7-4f9f-da14-3dc64e8f8626"
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df,how='right',left_on='personId',right_on='personId')

print('# of interactions from users with at least 5 interactions: {}'.format(len(interactions_from_selected_users_df)))
```

```python id="PyKJDhdPRxY6"
def preprocess_interactions(data):

  # define implicit signal strengths
  event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,
   }

   # quantify implicit signals based on corresponding strength
   data['eventStrength'] = data['eventType'].apply(lambda x: event_type_strength[x])

   # select users with at least n interactions
   _userids = data.groupby(['personId', 'contentId']).size().groupby('personId').size()
   _userids = _userids[_userids >= 5].reset_index()[['personId']]
  data = data.merge(_userids, how='right', on='personId')

  # apply log transformation to smooth the distribution
  data = data.groupby(['personId', 'contentId'])['eventStrength'].sum().apply(lambda x: math.log(1+x, 2)).reset_index()
```

```python id="O7V5UjZSXNHQ"
stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords_list)
```

```python id="dvcaa5E3RSrS"
def preprocess_items(data):

  # get only eventtype='CONTENT SHARED'
  data = data[data['eventType']=='CONTENT SHARED']
  
  # convert 'title' and 'text' column texts into tf-idf based numeric vectors
  tfidf_matrix = vectorizer.fit_transform(data['title'] + "" + data['text'])
```

<!-- #region id="HRhndICFYiql" -->
source - https://nbviewer.jupyter.org/github/amrqura/end-to-end-recommender-system/blob/main/notebook/recommenderSystemNotebook.ipynb
<!-- #endregion -->

```python id="10sLW6hUYjdG"

```
