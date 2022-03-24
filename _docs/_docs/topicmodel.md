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

<!-- #region id="iCE_tZl7ueIG" -->
# Topic Modeling on ServiceNow Dataset
<!-- #endregion -->

```python id="oqg7Pwgp2dom"
# !pip install pyLDAvis
# !pip install rake-nltk
```

```python id="fRfjyGqznR17"
path = '/content'
```

```python id="32ZuGdl9NTr_"
from nltk.util import ngrams
```

```python id="ycACxmXqm6FI" colab={"base_uri": "https://localhost:8080/", "height": 86} executionInfo={"status": "ok", "timestamp": 1601016105413, "user_tz": -330, "elapsed": 2265, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="45fa081d-3052-40ee-b7b4-7b2c71e2eee5"
import os
import pandas as pd

import re
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import groupby
from bs4 import BeautifulSoup
from collections import OrderedDict
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis

from rake_nltk import Rake

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
stopwords = list(set(stopwords.words('english')))

ps = PorterStemmer()
nltk.download('wordnet') 
lemmatizer = WordNetLemmatizer()
```

```python id="00zX_eAonLxj" colab={"base_uri": "https://localhost:8080/", "height": 156} executionInfo={"status": "ok", "timestamp": 1601015450499, "user_tz": -330, "elapsed": 1905, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="36575d4c-7339-4cab-ba20-98d2d4e7fd42"
files = os.listdir(path); files
```

```python id="qGZb-j1pnTlG"
Subset_Database = pd.read_excel(os.path.join(path,'Subset_Database.xlsx'))
Subset_Hardware = pd.read_excel(os.path.join(path,'Subset_Hardware.xlsx'))
Subset_Inquiry = pd.read_excel(os.path.join(path,'Subset_Inquiry.xlsx'))
Subset_Network = pd.read_excel(os.path.join(path,'Subset_Network.xlsx'))
Subset_Software = pd.read_excel(os.path.join(path,'Subset_Software.xlsx'))
```

```python id="8kzyUZcynO-z"
def clean_l1(text, extended=False):
  text = ' ' + text + ' '
  text = text.lower()
  text = re.sub(r'[^a-z ]', ' ', text)
  text = ' '.join(text.split())
  text = ' '.join([lemmatizer.lemmatize(w, 'v') for w in text.split()])
  text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
  if extended:
    text = ' '.join([w for w in text.split() if not w in stopwords])
  return text
```

```python id="7SRLwqcyq5yY"
def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(5, 5/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
```

```python id="ZZNzPlOl09z5"
def print_topics(model, count_vectorizer, n_top_words):
  words = count_vectorizer.get_feature_names()
  for topic_idx, topic in enumerate(model.components_):
      print("\nTopic #%d:" % topic_idx)
      print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
```

```python id="hD9GpzJZzjAo"
def topic_model(df, outfile='output'):
  # select short_desc column
  df = df[['Short description']]
  # clean
  df['Short description'] = df['Short description'].apply(clean_l1)
  # drop null and duplicates
  df = df.dropna().drop_duplicates()
  # drop records with 1 or less word
  df = df[df['Short description'].str.split().str.len().gt(1)]
  # compile list of documents
  docs = df['Short description'].tolist()
  # keyword extraction and tokenization
  min=2; max=2
  r = Rake(min_length=min, max_length=max)
  r.extract_keywords_from_sentences(docs)
  keywords = r.get_ranked_phrases()[:100]
  def tokenize(text):
    tokens = text.split()
    tokens = list(set(tokens))
    for i in range(2,max+1):
      ngram = list(ngrams(tokens, i))
      ngram = [' '.join(list(x)) for x in ngram]
      tokens.extend(ngram)
    tokens = list(set(tokens) & set(keywords))
    return tokens
  # top-k most common words
  count_vectorizer = CountVectorizer(stop_words='english', tokenizer=tokenize)
  count_data = count_vectorizer.fit_transform(docs)
  # plot_10_most_common_words(count_data, count_vectorizer)
  # generate wordcloud
  sum_words = count_data.sum(axis=0) 
  words_freq = [(word, sum_words[0, idx]) for word, idx in count_vectorizer.vocabulary_.items()]
  words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
  words_dict = dict(words_freq)
  wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
  wordcloud.generate_from_frequencies(words_dict)
  wordcloud.to_file(outfile+'.png')
  # generate topic model
  number_topics = 5 #PARAM
  number_words = 10 #PARAM
  lda = LDA(n_components=number_topics, n_jobs=-1)
  lda.fit(count_data)
  # print("Topics found via LDA:")
  # print_topics(lda, count_vectorizer, number_words)
  # LDA Visualization
  LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
  pyLDAvis.save_html(LDAvis_prepared, outfile+'.html')
```

```python id="3A5zyk7YnkdE"
topic_model(Subset_Database, 'Subset_Database')
topic_model(Subset_Hardware, 'Subset_Hardware')
topic_model(Subset_Inquiry, 'Subset_Inquiry')
topic_model(Subset_Network, 'Subset_Network')
topic_model(Subset_Software, 'Subset_Software')
```

```python id="rkO4_5EQJWks"

```
