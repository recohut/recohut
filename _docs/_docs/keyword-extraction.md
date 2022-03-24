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

<!-- #region id="pHW0olBoOyab" -->
# Keyword Extraction with RAKE and NLTK
<!-- #endregion -->

```python id="CuZOtdGTUqu5"
path = '/content'
```

```python id="8CNx-MhUVFZD"
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
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

```python id="GwUkY0fbVoNQ" colab={"base_uri": "https://localhost:8080/", "height": 51} executionInfo={"status": "ok", "timestamp": 1599451868063, "user_tz": -330, "elapsed": 1361, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c5462260-919f-4695-fc8c-bba8523b036b"
files = os.listdir(path); files
```

```python id="NqdKDWQPVIpm" colab={"base_uri": "https://localhost:8080/", "height": 221} executionInfo={"status": "ok", "timestamp": 1599451950584, "user_tz": -330, "elapsed": 1700, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="409f6348-48a4-4ca7-9ad4-7289700d5ad7"
df_raw = pd.read_excel(os.path.join(path,files[1]), index_col=[0])
df_raw.columns = ['domain','signup_url','other_lang','lang','title']
df_raw.info()
```

```python id="9FN9C2-LI_KV" colab={"base_uri": "https://localhost:8080/", "height": 51} executionInfo={"status": "ok", "timestamp": 1599458260903, "user_tz": -330, "elapsed": 1970, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="79f39919-26d1-4ffc-9c8c-d1d69c0b13e5"
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
```

```python id="qMZgqS_PXcMf"
def process(text):
  text = str(text)
  text = text.lower()
  text = re.sub(r'[^a-z0-9 ]','',text)
  text = ' '.join([lemmatizer.lemmatize(w, pos='v') for w in text.split()])
  text = ' '.join([lemmatizer.lemmatize(w, pos='n') for w in text.split()])
  text = ' '.join(text.split())
  return text
```

```python id="F1dXcs-aqWw4"
# xx = df_raw.sample(10, random_state=11)[['title']]
# xx = xx.dropna(subset=['title'])
# xx['title'] = xx['title'].apply(process)
# xx['title'] = xx['title'].replace(r'^\s*$', np.nan, regex=True)
# xx = xx.dropna(subset=['title'])
# xx
```

```python id="zHRFYPleVzYl" colab={"base_uri": "https://localhost:8080/", "height": 221} executionInfo={"status": "ok", "timestamp": 1599465007826, "user_tz": -330, "elapsed": 12065, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="73bd2614-d218-4b18-b077-4a862a5ac98c"
df = df_raw.copy()
df = df.dropna(subset=['title'])
df['title'] = df['title'].apply(process)
df['title'] = df['title'].replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(subset=['title'])
df.info()
```

```python id="m0fFhoEpJOH4" colab={"base_uri": "https://localhost:8080/", "height": 194} executionInfo={"status": "ok", "timestamp": 1599465007828, "user_tz": -330, "elapsed": 11837, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6a56ca3d-01d3-439f-be37-c4e8d10ab785"
df.sample(5)
```

```python id="qSnDov5pWWKu"
# !pip install langdetect
# from langdetect import detect
# langs = []
# df['lang'] = 'xx'
# for index, row in df.iterrows():
#   try:
#     df.loc[index,'lang'] = detect(row['title'])
#   except:
#     pass
# langs.append(detect(row['title']))
# pd.Series(langs).value_counts()
# df.to_csv('x.csv')
```

```python id="MUsImBvKkw9V"
def top_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df
    
def top_mean_feats(X, features, grp_ids=None, mins=0.1, top_n=25):
  if grp_ids:
      D = X[grp_ids].toarray()
  else:
      D = X.toarray()
  D[D < mins] = 0
  means = np.mean(D, axis=0)
  return top_feats(means, features, top_n)
```

```python id="CyyphwYM2NMW"
model_word_1gram = CountVectorizer(analyzer='word', stop_words='english', max_df=1.0, min_df=10, ngram_range=(1,1))
model_word_2gram = CountVectorizer(analyzer='word', stop_words=None, max_df=1.0, min_df=10, ngram_range=(2,2))
```

```python id="D3G6ilaly8cW"
matrix_word_1gram = model_word_1gram.fit_transform(df.title.tolist())
matrix_word_2gram = model_word_2gram.fit_transform(df.title.tolist())
```

```python id="GCXCBZm2zZzR"
features_word_1gram = model_word_1gram.get_feature_names()
features_word_2gram = model_word_2gram.get_feature_names()
```

```python id="UMq3Qo-u2QKf" colab={"base_uri": "https://localhost:8080/", "height": 374} executionInfo={"status": "ok", "timestamp": 1599467728341, "user_tz": -330, "elapsed": 1197, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98ea63f1-8476-475a-85d5-8ad53b8f21e3"
top_word_1gram = top_mean_feats(matrix_word_1gram, features_word_1gram, top_n=20)
print(top_word_1gram)
```

```python id="7E8k0sIu0cgQ" colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"status": "ok", "timestamp": 1599467873368, "user_tz": -330, "elapsed": 1125, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0a10189e-0013-4d7c-f60a-d66eb4fc9058"
top_word_2gram = top_mean_feats(matrix_word_2gram, features_word_2gram, top_n=20)
print(top_word_2gram)
```

```python id="WZEP7l8u2D08"
from nltk.util import ngrams
def create_chargrams(word):
  gramslist = []
  for n in range(2,6):
    grams = list(ngrams(list(word), n))
    grams = list(set(list(map(''.join, grams))))
    gramslist.extend(grams)
  return gramslist

def chargram_similarity(word1, word2):
  gramlist1 = create_chargrams(word1)
  gramlist2 = create_chargrams(word2)
  gramlistc = list(set(gramlist1).intersection(gramlist2))
  simscore = len(gramlistc)/min(len(gramlist1), len(gramlist2))
  return simscore
```

```python id="pd2Tqvw23_kG" colab={"base_uri": "https://localhost:8080/", "height": 68} executionInfo={"status": "ok", "timestamp": 1599468378976, "user_tz": -330, "elapsed": 1221, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="63fc49f2-5d98-4b22-f20d-834484afdee5"
print("('signin','sign in'): ", chargram_similarity('signin','sign in'))
print("('signin','create'): ", chargram_similarity('signin','create'))
print("('signin','signup'): ", chargram_similarity('signin','signup'))
```

```python id="1ut8j6h2A34T" colab={"base_uri": "https://localhost:8080/", "height": 187} executionInfo={"status": "ok", "timestamp": 1599465071465, "user_tz": -330, "elapsed": 4043, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7bcf1c52-d942-4aa5-91de-ce21d6a95b5f"
# !pip install pytextrank
# !python -m spacy download en_core_web_sm
# import spacy
# import pytextrank
text = ' . '.join(df.title.tolist())
nlp = spacy.load("en_core_web_sm")
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
doc = nlp(text)
for p in doc._.phrases[:10]:
    # print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text)); print(p.chunks)
    print("{:.4f} {}".format(p.rank, p.text))
```

```python id="r2C0y-PX1ZvH" colab={"base_uri": "https://localhost:8080/", "height": 187} executionInfo={"status": "ok", "timestamp": 1599468550265, "user_tz": -330, "elapsed": 15566, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e2994fd1-7e23-4428-b323-cc13136e0003"
# !pip install summa
# from summa import keywords
# text = ' . '.join(df.title.tolist())
xx = keywords.keywords(text, scores=True, deaccent=True)
xx[:10]
```

```python id="zIStEoUFkcQw" colab={"base_uri": "https://localhost:8080/", "height": 187} executionInfo={"status": "ok", "timestamp": 1599465976713, "user_tz": -330, "elapsed": 2346, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="afc4fdf5-1be3-4cce-e40e-c4fabb7a3d79"
# !pip install rake-nltk
# from rake_nltk import Metric, Rake
# rake = Rake(min_length=1, max_length=2, ranking_metric=Metric.WORD_FREQUENCY)
# text = ' . '.join(df.title.tolist())
# rake.extract_keywords_from_text(text)
rake.get_ranked_phrases_with_scores()[:10]
```

```python id="gi30f6sJmyOk" colab={"base_uri": "https://localhost:8080/", "height": 187} executionInfo={"status": "ok", "timestamp": 1599466073290, "user_tz": -330, "elapsed": 951, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b65955f0-8565-47b3-cf92-f81a22e28d69"
rake = Rake(min_length=1, max_length=1, ranking_metric=Metric.WORD_FREQUENCY)
rake.extract_keywords_from_text(text)
rake.get_ranked_phrases_with_scores()[:10]
```

<!-- #region id="m-hloyO9zx11" -->
---
<!-- #endregion -->

```python id="oLRSdOGmY8DV"
# df_tfidf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
# print(df_tfidf)

# list(tfidf.vocabulary_.keys())[:10]

# df_tfidf = pd.DataFrame(x[0].T.todense(), index=tfidf.get_feature_names(), columns=["TF-IDF"])
# df_tfidf = df_tfidf.sort_values('TF-IDF', ascending=False)
# print(df_tfidf.head(25))
```
