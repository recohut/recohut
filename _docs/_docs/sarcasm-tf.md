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

<!-- #region id="HBF2okXCVjmw" -->
# Sarcasm Detection in Tensorflow
<!-- #endregion -->

```python id="kuk7dT6r_vWc"
path = '/content/drive/My Drive/Playground/DAF0AC92369C4F74A4AAA2AE089DFDB2'
```

```python id="q3o9tg-NAZSU"
!unzip '/content/drive/My Drive/Playground/DAF0AC92369C4F74A4AAA2AE089DFDB2/Sarcasm_Headlines_Dataset_v2.zip'
```

```python id="N47CVoQTAgdg"
import os
import re
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import KeyedVectors

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

```python id="bcCadMJZYp24"
import tensorflow as tf
import tensorflow.keras as keras 
from keras.models import Sequential, Model 
from keras import layers
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
```

```python id="MHXMcep9BN8D" colab={"base_uri": "https://localhost:8080/", "height": 274} executionInfo={"status": "ok", "timestamp": 1596285517595, "user_tz": -330, "elapsed": 1036, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="52791d7c-e721-4e69-b3e2-00b5ccb580d0"
def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

data = list(parse_data('Sarcasm_Headlines_Dataset_v2.json'))
df = pd.DataFrame(data)
df.head()
```

```python id="ZDRdFV7zX2CX" colab={"base_uri": "https://localhost:8080/", "height": 68} executionInfo={"status": "ok", "timestamp": 1596291258441, "user_tz": -330, "elapsed": 621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="524468fa-73ca-42a1-a8b1-7a98bbc6dd6c"
df.is_sarcastic.value_counts()
```

```python id="t2hsa7nsBZ96"
def clean_text(corpus):
  cleaned_corpus = pd.Series()
  for row in corpus:
      qs = []
      for word in row.split():
          p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
          p1 = p1.lower()
          qs.append(p1)
      cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
  return cleaned_corpus

def stopwords_removal(corpus):
    stop = set(stopwords.words('english'))
    corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    return corpus
  
def lemmatize(corpus):
    lem = WordNetLemmatizer()
    corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    return corpus

def stem(corpus, stem_type = None):
    if stem_type == 'snowball':
        stemmer = SnowballStemmer(language = 'english')
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    else :
        stemmer = PorterStemmer()
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    return corpus

def preprocess(corpus, cleaning = True, stemming = False, stem_type = None, lemmatization = False, remove_stopwords = True):
    if cleaning:
        corpus = clean_text(corpus)
    if remove_stopwords:
        corpus = stopwords_removal(corpus)
    else:
        corpus = [[x for x in x.split()] for x in corpus]
    if lemmatization:
        corpus = lemmatize(corpus)
    if stemming == True:
        corpus = stem(corpus, stem_type)
    corpus = [' '.join(x) for x in corpus]
    return corpus
```

```python id="sVdI4HPlMQ05"
headlines = preprocess(df['headline'], lemmatization = True, remove_stopwords = True)
```

```python id="ykAnD74LNPil"
# !wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
```

```python id="KJqTd_9NUnZU"
MAX_LENGTH = 10
VECTOR_SIZE = 300

def vectorize_data(data):
  vectors = []
  padding_vector = [0.0] * VECTOR_SIZE
  for i, data_point in enumerate(data):
    data_point_vectors = []
    count = 0
    tokens = data_point.split()
    for token in tokens:
      if count >= MAX_LENGTH:
        break
      if token in model.wv.vocab:
        data_point_vectors.append(model.wv[token])
      count+=1
    if len(data_point_vectors) < MAX_LENGTH:
      to_fill = MAX_LENGTH - len(data_point_vectors)
      for _ in range(to_fill):
        data_point_vectors.append(padding_vector)
    vectors.append(data_point_vectors)
  return vectors
```

```python id="fQtDJFOEXXSz"
vectorized_headlines = vectorize_data(headlines)
```

```python id="FpZHrheZXscw" colab={"base_uri": "https://localhost:8080/", "height": 85} executionInfo={"status": "ok", "timestamp": 1596291394447, "user_tz": -330, "elapsed": 2363, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="76c315c5-2580-4234-bfca-4891dad600e3"
train_div = math.floor(0.7 * len(vectorized_headlines))

X_train = vectorized_headlines[:train_div]
y_train = df['is_sarcastic'][:train_div]
X_test = vectorized_headlines[train_div:]
y_test = df['is_sarcastic'][train_div:]

print('The size of X_train is:', len(X_train), '\nThe size of y_train is:', len(y_train),
      '\nThe size of X_test is:', len(X_test), '\nThe size of y_test is:', len(y_test))

X_train = np.reshape(X_train, (len(X_train), MAX_LENGTH, VECTOR_SIZE))
X_test = np.reshape(X_test, (len(X_test), MAX_LENGTH, VECTOR_SIZE))
y_train = np.array(y_train)
y_test = np.array(y_test)
```

```python id="wOkQbMnNYWQb"
FILTERS=8
KERNEL_SIZE=3
HIDDEN_LAYER_1_NODES=10
HIDDEN_LAYER_2_NODES=5
DROPOUT_PROB=0.35
NUM_EPOCHS=10
BATCH_SIZE=50
```

```python id="sx7xixBXYc27" colab={"base_uri": "https://localhost:8080/", "height": 408} executionInfo={"status": "ok", "timestamp": 1596291498917, "user_tz": -330, "elapsed": 1332, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="51e19883-12c6-4fd2-d3d7-bd36d5ec7546"
model = Sequential()

model.add(Conv1D(FILTERS,
                 KERNEL_SIZE,
                 padding='same',
                 strides=1,
                 activation='relu', 
                 input_shape = (MAX_LENGTH, VECTOR_SIZE)))
model.add(GlobalMaxPooling1D())
model.add(Dense(HIDDEN_LAYER_1_NODES, activation='relu'))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(HIDDEN_LAYER_2_NODES, activation='relu'))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
```

```python id="uUP9uBLZYk5x"
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

```python id="cRkhMFnGY2hg" colab={"base_uri": "https://localhost:8080/", "height": 357} executionInfo={"status": "ok", "timestamp": 1596291543542, "user_tz": -330, "elapsed": 25614, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e8e28a44-6946-4982-a852-3536e5e6b30d"
training_history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
```

```python id="VYOJMjxrY41d" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1596291557432, "user_tz": -330, "elapsed": 1539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="485bbc10-f40e-4e1d-93b6-ca47e8680e4d"
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
```

```python id="VexECTPzZCGW"
model_structure = model.to_json()
with open("sarcasm_detection_model_cnn.json", "w") as json_file:
    json_file.write(model_structure)
model.save_weights("sarcasm_detection_model_cnn.h5")
```
