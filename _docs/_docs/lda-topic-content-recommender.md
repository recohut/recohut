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

<!-- #region id="tYcrtIAuyRhd" -->
# Similar Document Recommendations using Topic Modeling
> Training LDA topic model on 20-newsgroups dataset and finding similar documents using KNN

- toc: true
- badges: true
- comments: true
- categories: [LDA, Topic Modeling, Ktrain, KNN]
- image:
<!-- #endregion -->

<!-- #region id="iKVRDDqxpFzt" -->
### Setup
<!-- #endregion -->

```python id="DGM1G0ONj4JQ"
!pip install ktrain
!pip install stellargraph 
```

```python id="UuPe3RG2j6HB"
import os
import numpy as np
import pandas as pd

import ktrain

pd.set_option('display.max_colwidth', -1)
%matplotlib inline
```

<!-- #region id="7eBinb2BpHUq" -->
Using LDA Topic modeling technique to find similar documents.
<!-- #endregion -->

<!-- #region id="CXyEiUW1pk6H" -->
### What is Topic Model?
<!-- #endregion -->

<!-- #region id="w1hr9W0XpzA4" -->
<!-- #endregion -->

<!-- #region id="NMmko2nDpqHZ" -->
In natural language processing, the term topic means a set of words that “go together”. These are the words that come to mind when thinking of this topic. Take sports. Some such words are athlete, soccer, and stadium.

A topic model is one that automatically discovers topics occurring in a collection of documents.
<!-- #endregion -->

<!-- #region id="F3UwnMuQpwPL" -->
<!-- #endregion -->

<!-- #region id="_CbbDq1AneXh" -->
### Get Raw Document Data
<!-- #endregion -->

<!-- #region id="NdkluUD7pgSQ" -->
A collection of ~18,000 newsgroup documents from 20 different newsgroups
<!-- #endregion -->

<!-- #region id="HDMG9JQJpZRw" -->
<!-- #endregion -->

```python id="_O4gqaNOkIt0"
# 20newsgroups
from sklearn.datasets import fetch_20newsgroups
remove = ('headers', 'footers', 'quotes')
newsgroups_train = fetch_20newsgroups(subset='train', remove=remove)
newsgroups_test = fetch_20newsgroups(subset='test', remove=remove)
texts = newsgroups_train.data +  newsgroups_test.data
```

```python colab={"base_uri": "https://localhost:8080/"} id="cVcHmNyukTov" outputId="c6283855-b0c9-4d3d-e869-825ebd08016d"
texts[:10]
```

<!-- #region id="O-xcM18CnhVx" -->
### Represent Documents as Semantically Meaningful Vectors With LDA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WLRd5tpekRue" outputId="5867b078-a504-416c-d7c1-1bfd7e2fc895"
%%time
tm = ktrain.text.get_topic_model(texts, n_features=10000)
```

```python colab={"base_uri": "https://localhost:8080/"} id="j06Appkct9hj" outputId="f3035091-92bc-4576-b347-dcc614b083d6"
help(tm.build)
```

```python colab={"base_uri": "https://localhost:8080/"} id="mR4VPze2nlfg" outputId="ca273893-6b05-48e3-ad14-7939d4cd3c3d"
%%time
tm.build(texts, threshold=0.25)
```

<!-- #region id="3Q3Su7i6nqOL" -->
### Train a Document Recommender
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="rxFN036YvD4v" outputId="19ca6d0e-5909-4e7e-c46f-a3a7bae189d4"
help(tm.train_recommender)
```

```python id="JaHg3Aghnqzq"
tm.train_recommender()
```

<!-- #region id="wOUrLTzQnuG1" -->
### Generate Recommendations
<!-- #endregion -->

<!-- #region id="Jwc-zU7wnw0w" -->
Given some text, recommend documents that are semantically relevant to it.
<!-- #endregion -->

```python id="ulWIIEkNnuhy"
rawtext = """
            Elon Musk leads Space Exploration Technologies (SpaceX), where he oversees
            the development and manufacturing of advanced rockets and spacecraft for missions
            to and beyond Earth orbit.
            """
```

```python colab={"base_uri": "https://localhost:8080/"} id="4VqENw40n2yN" outputId="7093c664-e135-49d4-ea8d-bee94e8f583e"
for i, doc in enumerate(tm.recommend(text=rawtext, n=5)):
    print('RESULT #%s'% (i+1))
    print('TEXT:\n\t%s' % (" ".join(doc['text'].split()[:500])))
    print()
```

<!-- #region id="PD3eNF7on6wR" -->
### Saving and Restoring the Topic Model
The topic model can be saved and restored as follows.
<!-- #endregion -->

```python id="gt1zkvlzn8jL"
tm.save('/content/tm')
```

```python colab={"base_uri": "https://localhost:8080/"} id="vghV5OAToAgl" outputId="d6696ae2-c213-41fb-d5e9-1aa5fb20d9c3"
tm = ktrain.text.load_topic_model('/content/tm')
tm.build(texts, threshold=0.25)
```

<!-- #region id="yZRWBaY6oMVF" -->
> Note: the scorer and recommender are not saved, only the LDA topic model is saved. So, the scorer and recommender should be retrained prior to use
<!-- #endregion -->

```python id="hQ2oMh4xoO6P"
tm.train_recommender()
```

```python id="wfTAnUEBoXDn"
rawtext = """
            Elon Musk leads Space Exploration Technologies (SpaceX), where he oversees
            the development and manufacturing of advanced rockets and spacecraft for missions
            to and beyond Earth orbit.
            """
```

```python colab={"base_uri": "https://localhost:8080/"} id="Gdr87_6koXcW" outputId="cfa322be-59f5-43d3-ad0c-0069302c2ef8"
#collapse-hide
print(tm.recommend(text=rawtext, n=1)[0]['text'])
```
