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

<!-- #region id="EF4jikZn8Vxd" -->
# Job scraping and clustering
> Simple web scraping with BeautifulSoup4 and NLP with DistilBERT 

- toc: true
- badges: true
- comments: true
- categories: [scraping, bert]
- image: 
<!-- #endregion -->

<!-- #region id="RjzTML0XaBxB" -->
## Part 1 - Environment Setup
<!-- #endregion -->

```python id="wHTCNnba06md"
!pip install -q requests beautifulsoup4
!pip install -U sentence-transformers
```

```python id="00HKz3Is0_me" colab={"base_uri": "https://localhost:8080/"} outputId="74ee0a26-980a-4d72-ff85-fa37a50d266e"
import time
import csv
import re

import numpy as np
import pandas as pd
import requests
import bs4
import lxml.etree as xml

import pprint
from scipy.spatial.distance import cosine, cdist

import nltk
nltk.download('punkt')

from spacy.lang.en import English
nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

%reload_ext google.colab.data_table
```

```python id="Ag1EdGqC1I0u" colab={"base_uri": "https://localhost:8080/"} outputId="03f53658-c355-4175-ac4c-2930ae0bbced"
URLs = ["https://www.flexjobs.com/blog/post/job-search-strategies-for-success-v2/",
        "https://www.best-job-interview.com/job-search-strategy.html",
        "https://content.wisestep.com/job-search-strategies/",
        "https://www.thebalancecareers.com/top-strategies-for-a-successful-job-search-2060714",
        "https://www.monster.com/career-advice/article/a-winning-job-search-strategy",
        "https://interviewdoctor.com/testimonials/",
        "https://www.telenor.com/10-tips-for-job-hunting-in-the-digital-age/",
        "https://www.monster.com/career-advice/article/five-ps-of-job-search-progress",
        ]

requests.get(URLs[7])
```

<!-- #region id="Da3VLP77aKaA" -->
## Part 2 - Scraping
<!-- #endregion -->

```python id="FvrmHyJHXg54"
df = pd.DataFrame(columns=['title','text'])
```

```python id="VanjjiMLANOy"
i = 0
web_page = bs4.BeautifulSoup(requests.get(URLs[i], {}).text, "lxml")
df.loc[i,'title'] = web_page.head.title.text
sub_web_page = web_page.find_all(name="article", attrs={"class": "single-post-page"})[0]
article = '. '.join([wp.text for wp in sub_web_page.find_all({"h2","p"})])
df.loc[i,'text'] = article
```

```python id="8QlSMcKTAZg7"
i = 1
web_page = bs4.BeautifulSoup(requests.get(URLs[i], {}).text, "lxml")
df.loc[i,'title'] = web_page.head.title.text
sub_web_page = web_page.find_all(attrs={"id": "ContentColumn"})[0]
article = '. '.join([wp.text for wp in sub_web_page.find_all({"span","h2","p"})])
df.loc[i,'text'] = article
```

```python id="xTQW5t4TEk3M"
i = 2
web_page = bs4.BeautifulSoup(requests.get(URLs[i], {}).text, "lxml")
df.loc[i,'title'] = web_page.head.title.text
sub_web_page = web_page.find_all(attrs={"class": "td-ss-main-content"})[0]
article = '. '.join([wp.text for wp in sub_web_page.find_all({"span","h2","p"})])
df.loc[i,'text'] = article
```

```python id="stoQ9rx1KITk"
i = 3
web_page = bs4.BeautifulSoup(requests.get(URLs[i], {}).text, "lxml")
df.loc[i,'title'] = web_page.head.title.text
sub_web_page = web_page.find_all(attrs={"id": "list-sc_1-0"})[0]
article = '. '.join([wp.text for wp in sub_web_page.find_all({"h2","p"})])
df.loc[i,'text'] = article
```

```python id="6gzWk3g2Goi5"
i = 4
web_page = bs4.BeautifulSoup(requests.get(URLs[i], {}).text, "lxml")
df.loc[i,'title'] = web_page.head.title.text
sub_web_page = web_page.find_all(attrs={"id": "mainContent"})[0]
article = '. '.join([wp.text for wp in sub_web_page.find_all({"h2","p"})])
df.loc[i,'text'] = article
```

```python id="Dk4uOkZcMQEw"
i = 5
web_page = bs4.BeautifulSoup(requests.get(URLs[i], {}).text, "lxml")
df.loc[i,'title'] = web_page.head.title.text
sub_web_page = web_page.find_all(attrs={"class": "site-inner"})[0]
article = '. '.join([wp.text for wp in sub_web_page.find_all({"blockquote"})])
df.loc[i,'text'] = article
```

```python id="NdzVb-1MOwN4"
i = 6
web_page = bs4.BeautifulSoup(requests.get(URLs[i], {}).text, "lxml")
df.loc[i,'title'] = web_page.head.title.text
sub_web_page = web_page.find_all(attrs={"id": "primary"})[0]
article = '. '.join([wp.text for wp in sub_web_page.find_all({"p","ol"})])
df.loc[i,'text'] = article
```

```python id="BzqvrVb8RwvM"
i = 7
web_page = bs4.BeautifulSoup(requests.get(URLs[i], {}).text, "lxml")
df.loc[i,'title'] = web_page.head.title.text
sub_web_page = web_page.find_all(attrs={"class": "article-content"})[0]
article = '. '.join([wp.text for wp in sub_web_page.find_all({"p","h2"})])
df.loc[i,'text'] = article
```

```python id="jRzm2wCqOzkb" colab={"base_uri": "https://localhost:8080/"} outputId="06849979-bc50-42db-9969-4ed45e85a996"
df = df.dropna().reset_index(drop=True)
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="dHPGx4SzYi26" outputId="f9afed13-6e79-449f-a740-b528323149ea"
df
```

<!-- #region id="1tbb-hABaP8V" -->
## Part 3 - Text Preprocessing
<!-- #endregion -->

```python id="5qgGGZFwXD-m"
def tokenize(x):
  return nltk.sent_tokenize(x)
```

```python id="gF5EQ_neRg4V"
def spacy_tokenize(x):
  doc = nlp(x)
  return list(doc.sents)
```

```python id="lx-93M5ZeHtA"
def sentenize(temp, col = 'text'):
  s = temp.apply(lambda x: pd.Series(x[col]),axis=1).stack().reset_index(level=1, drop=True)
  s.name = col
  temp = temp.drop(col, axis=1).join(s)
  return temp
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="eMa-nBckXowB" outputId="8041e345-8771-480b-821f-9a87e11251ca"
temp = df[['text']].copy()

temp.loc[:,'text'] = temp.text.apply(lambda x: re.sub(r'\.+', ".", x))

temp.loc[:,'text'] = temp['text'].apply(tokenize)
temp = sentenize(temp,'text')
temp.reset_index(inplace=True)
temp.columns = ['para_id','text']

temp.loc[:,'text'] = temp['text'].apply(spacy_tokenize)
temp = sentenize(temp,'text')
temp.reset_index(drop=True, inplace=True)

temp = temp.dropna()

temp.loc[:,'text'] = temp.text.apply(lambda x: x.text.lower())

temp.loc[:,'text'] = temp['text'].str.replace("[^a-zA-Z0-9]", " ")

temp.loc[:,'text'] = temp['text'].dropna()

temp = temp[temp['text'].str.split().str.len().gt(3)]

temp = temp.drop_duplicates(subset=['text'], keep='first')

temp = temp.reset_index(drop=True)

temp
```

<!-- #region id="i7-T48Ataej9" -->
## Part 4 - Text clustering using distilbert
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="KSBM_wkgZeEp" outputId="9fead659-e02a-4105-b3fa-dfc68a9234f6"
embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
corpus = temp.text.tolist()
corpus_embeddings = embedder.encode(corpus)
```

```python colab={"base_uri": "https://localhost:8080/"} id="oVOXz_pMZena" outputId="4a925590-b625-4b98-b596-f8b547d66ee2"
queries = ['customize resume']
query_embeddings = embedder.encode(queries)
for query, query_embedding in zip(queries, query_embeddings):
    distances = cdist([query_embedding], corpus_embeddings, "cosine")[0]
    topn_index = distances.argsort()[:5][::-1]
    print('Query:', query)
    print('Top 5 most similar sentences in corpus:')
    for i in topn_index:
      pprint.pprint("{} (Score: {})".format(corpus[i], distances[i]))
```

```python id="y0dtT1jBZmMC"
num_clusters = 20
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="mkrQFoSVZsvX" outputId="23b1f50a-1b1e-4f0a-f0fb-4c3f0a068e2c"
df = pd.DataFrame(data={"text":corpus, "cluster":cluster_assignment})
df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 793} id="3caL-UitZssz" outputId="eebf3d59-a58b-449f-f838-0d4dd4eac965"
c = 0
df.loc[df.cluster==c,:]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="bEqryxqyZspl" outputId="e5eae5ef-e7bb-4058-e8aa-b43392dfa18f"
c = 1
df.loc[df.cluster==c,:]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 886} id="_PSr77HOZsmx" outputId="509d34eb-fa5b-4b92-ac3a-fa7ac8a6e2c4"
c = 6
df.loc[df.cluster==c,:]
```
