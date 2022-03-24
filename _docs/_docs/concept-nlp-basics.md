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
    language: python
    name: python3
---

<!-- #region id="v8RREnf5Ms_I" -->
# Concept - Natural Language Processing 101
> Understand the basic process of pattern recognition that is needed to work with text data. This approach can be used for a large number of basic operations on text data like Document Classification (e.g. topic of an article),Sequence to Sequence Learning (e.g. translations), and Sentiment Analysis

- toc: true
- badges: true
- comments: true
- categories: [Concept, TFIDF, NLP, Visualization, Altair]
- image:
<!-- #endregion -->

<!-- #region id="aX0HRp2SM6c2" -->
Lets understand the basic process of pattern recognition that is needed to work with text data. This approach can be used for a large number of basic operations on text data like 
- Document Classification (e.g. topic of an article)
- Sequence to Sequence Learning (e.g. translations)
- Sentiment Analysis

Lets start with seeing how a text sequence can be encoded into numbers and processed to prepare for these tasks
<!-- #endregion -->

<!-- #region id="KrAAWGSAMs_a" -->
## Tokenisation

Lets take the sentence - **The quick brown fox jumped over the lazy dog**

We need to first break this sentence in to smaller constituents - called **tokens**. Now there are three ways of creating the tokens can happen:

- **Individual character** - create tokens for each
- **Individual word** - Create tokens for each word in the sentence
- **N-gram** - Create tokens by taking n-grams words in the sentence
<!-- #endregion -->

<!-- #region id="_247UN-8Ms_c" -->
### Create word tokens
<!-- #endregion -->

<!-- #region id="ra093bFgMs_d" -->
**Pre-processing - split, punctuation & case**

There is some basic **pre-processing** that has been done in the process of creating word tokens
- Split the sentence on **whitespace**
- Filter **punctuations**
- Change to **lower case** text

After this pre-processing, we can get each token.
<!-- #endregion -->

```python id="CkczbwgEMs_f"
import numpy as np
import pandas as pd
import altair as alt

import spacy
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot, hashing_trick
```

```python id="MLUmP61sMs_v"
! python -m spacy download en_core_web_sm
```

```python id="CGFKq3l_Ms_k"
sentence = 'The quick brown fox jumped over the lazy dog.'
```

```python id="f-MWZVh8Ms_l" outputId="7351306b-e5fb-4fd8-9492-4def61dad5a6" colab={"base_uri": "https://localhost:8080/"}
text_to_word_sequence(sentence)
```

```python id="itGv8WC2Ms_x"
nlp = spacy.load('en_core_web_sm')
```

```python id="JPvnhJ_-Ms_z"
sentence = 'The quick brown fox jumped over the lazy dog'
doc = nlp(sentence)
```

```python id="IRo0m0kiMs_0" outputId="0a13b9b4-6dfd-4e3c-b699-1ab89bacb49d" colab={"base_uri": "https://localhost:8080/"}
for token in doc:
    print(token)
```

<!-- #region id="6dKO21nPMs_3" -->
## Vectorisation

Once you have tokens, we need to find a way to represent them as vectors. Let's look at two traditional way of representing them as vectors

- Frequency Based
    - Binary
    - Count 
    - tfidf
    - Co-occurence (Skipgram)
- Prediction Based
    - Pre-trained Vectors
    - Learning Vectors
    - Learning vectors with the task
<!-- #endregion -->

<!-- #region id="fBLXilXtMs_5" -->
### One-Hot Encoding 
<!-- #endregion -->

```python id="jPf81n4XMs_7" outputId="170eacc7-62ff-4d6c-f653-016d2bc8b238" colab={"base_uri": "https://localhost:8080/"}
# Given a size of vocabulary, do one-hot encoding
one_hot(sentence, n=10)
```

```python id="NdxhaclNMs_9" outputId="8ccf6bf4-064a-464d-afdd-3c53f1c6dc19" colab={"base_uri": "https://localhost:8080/"}
# Given a size of vocabulary, do hash encoding (to save space)
hashing_trick(sentence, n=100, hash_function="md5")
```

<!-- #region id="yhzRPwUxMs_-" -->
> Tip: Using the Tokenizer API
<!-- #endregion -->

```python id="tP4D3xYbMtAA"
from keras.preprocessing.text import Tokenizer
```

```python id="9xSSfmK0MtAB"
# Instantiate the Tokenizer
simple_tokenizer = Tokenizer()
```

```python id="nDjdGU48MtAD"
# Fit the Tokenizer
simple_tokenizer.fit_on_texts([sentence])
```

```python id="4rs3IUXyMtAU"
def get_sentence_vectors(sentences, tokenizer, mode="binary"):
    matrix = tokenizer.texts_to_matrix(sentences, mode=mode)
    df = pd.DataFrame(matrix)
    df.drop(columns=0, inplace=True)
    df.columns = tokenizer.word_index
    return df
```

```python id="oCzgQYTYMtAE" outputId="bad34f45-bf44-482b-bd54-edfb368db5ab" colab={"base_uri": "https://localhost:8080/"}
# See the word vectors
simple_tokenizer.word_index
```

<!-- #region id="kiJjAcjiMtAI" -->
Normally we will be working with a set of text (like sentences), so it is better to use the tokenizer API
<!-- #endregion -->

```python id="zaJAHfcTMtAK"
sentences = ['The quick brown fox jumped over the lazy dog', 
             'The dog woke up lazily and barked at the fox',
             'The fox looked back and just ignored the dog']
```

```python id="LZ_dESr1MtAL"
# Instantiate and Fit
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
```

```python id="95uSs146RREx" outputId="d93f0b4f-48ff-4bf2-b73b-4c70d4d78c06" colab={"base_uri": "https://localhost:8080/"}
tokenizer.word_index
```

```python id="-NOA9LArMtAN" outputId="9e43af8f-9583-4130-ed0d-d83314c17526" colab={"base_uri": "https://localhost:8080/"}
tokenizer.texts_to_sequences(sentences)
```

```python id="d5j3-XLnMtAP" outputId="f0a57191-e564-4d98-b44e-6d1a401647af" colab={"base_uri": "https://localhost:8080/"}
tokenizer.texts_to_matrix(sentences, mode="binary")
```

```python id="jgHR2qghMtAV" outputId="676773cf-e5f4-4c34-ada8-a46b496bf36f" colab={"base_uri": "https://localhost:8080/", "height": 162}
_x = get_sentence_vectors(sentences, tokenizer, mode="tfidf")
_x
```

```python id="a0xngigzVyod" outputId="0c6e2134-0e84-4f41-e248-ad978d2580d0" colab={"base_uri": "https://localhost:8080/", "height": 359}
_x = _x.rename_axis('sentence').reset_index().melt(id_vars=['sentence'])
_x.head(10)
```

```python id="RpAjw1BJMtAY" outputId="322fbf82-3560-4434-97c4-a572ad8713ad" colab={"base_uri": "https://localhost:8080/", "height": 147}
alt.Chart(_x).mark_rect().encode(
    x=alt.X('variable:N', title="word"),
    y=alt.Y('sentence:N', title="sentence"),
    color=alt.Color('value:Q', title="tfidf")
).properties(
    width=700
).interactive()
```

```python id="FxSiN_LsMtAb" outputId="0186d431-6704-4d64-93f6-a7514db1f1a9" colab={"base_uri": "https://localhost:8080/"}
one_hot_results = tokenizer.texts_to_matrix(sentence, mode='binary')

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
```
