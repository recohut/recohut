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

```python id="QHfJAkdQk3jv"
import os
project_name = "reco-tut-vrr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3APW7y3clBlH" executionInfo={"status": "ok", "timestamp": 1628408434630, "user_tz": -330, "elapsed": 1378, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="17900773-0b16-41fd-c1d3-57d20c5a748b"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="yGo_LJd6lBlN"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="SWzR_TDJlBlO" executionInfo={"status": "ok", "timestamp": 1628408390337, "user_tz": -330, "elapsed": 2087, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1f0aa5a6-f8e0-4bd1-8b83-cea3865fa6dc"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="ra084G6QmDcL" -->
---
<!-- #endregion -->

```python id="Nrb7EyBKmEmk"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras import models
from keras.layers.embeddings import Embedding

import warnings
warnings.filterwarnings('ignore')
```

```python colab={"base_uri": "https://localhost:8080/"} id="ce393Lpposh_" executionInfo={"status": "ok", "timestamp": 1628408439309, "user_tz": -330, "elapsed": 714, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2db10727-181b-4a94-bc29-e6d5422a74cf"
nltk.download('wordnet')
nltk.download('stopwords')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="HF4J7q62mJ7u" executionInfo={"status": "ok", "timestamp": 1628408460344, "user_tz": -330, "elapsed": 832, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="deeca935-07f5-4b92-9835-da83411264d5"
df = pd.read_parquet('./data/silver/reviews.parquet.gzip')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="knyLYsCEmT0d" executionInfo={"status": "ok", "timestamp": 1628408464417, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d600bbe4-a1f2-4401-8f92-ede0b17210db"
df.info()
```

```python id="UdcHVDdS0BOx"
#filter punctuations, stemming and stopwords
corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z0-9]', ' ', df['Reviews'][i])
    review = review.lower()
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stemmer = nltk.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
    tokens_list = tokenizer.tokenize(review)
    tokens = []
    for token in tokens_list:
        tokens.append(lemmatizer.lemmatize(token))
        stop_words = stopwords.words("english")
    filtered_words = [w for w in tokens if w not in stop_words]
    review = ' '.join(filtered_words)
    corpus.append(review)
```

```python colab={"base_uri": "https://localhost:8080/"} id="pe1b_dKWop6y" executionInfo={"status": "ok", "timestamp": 1628410285697, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6dd4296d-c2de-4146-cde5-d4876f6510d1"
print('length of coprus is {} and first item is "{}"'.format(len(corpus), corpus[0]))
```

```python id="F8vG-YKC0Fal"
#Bag of Words model to convert corpus into X
cv = CountVectorizer()
cv.fit(corpus)
key = list(cv.vocabulary_.keys())
key.sort()
X = pd.DataFrame(cv.transform(corpus).toarray(),columns = key)
y = df.Rating
```

```python id="MdLG9_a_0JIv"
#TF_IDF model to convert corpus into X
tfidf = TfidfVectorizer()
X2 = pd.DataFrame(tfidf.fit_transform(corpus).toarray())
```

```python id="Jo4ZW6-w1cQG"
Rating = df.Rating
```

```python colab={"base_uri": "https://localhost:8080/"} id="r8YKopCu0YiU" executionInfo={"status": "ok", "timestamp": 1628410321928, "user_tz": -330, "elapsed": 911, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="12db105a-f12a-4397-d031-436b6326d278"
# We need to get unique words to determine the vocabulary size
reviews = df.Reviews
uniq_words=set()

for doc in reviews:
    for word in doc.split(" "):
        uniq_words.add(word)
vocab_size=len(uniq_words)

print ("Total Unique words:",vocab_size)
```

<!-- #region id="gRpPNRWAy16I" -->
## Review text to rating prediction
<!-- #endregion -->

<!-- #region id="WAZX4K6x0oK_" -->
We need to convert each of the words in the reviews to one-hot vectors. Below is the code to get integer indexes of the words for one hot vector.

Note that we don't need to store all zeros as only the integer index for the word in a vector will have a value of 1.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="O0mbja6W0t_5" executionInfo={"status": "ok", "timestamp": 1628410407382, "user_tz": -330, "elapsed": 520, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="640d4589-6639-4e4a-ae7a-3f7a15dfd75e"
# Integer encode the documents
encoded_reviews = [one_hot(review, vocab_size) for review in reviews]
print(encoded_reviews[1])
```

```python colab={"base_uri": "https://localhost:8080/"} id="0c6xxGn00wES" executionInfo={"status": "ok", "timestamp": 1628410416732, "user_tz": -330, "elapsed": 799, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="545e1714-5ef0-4d6d-b7a5-3a77fff0b55d"
# We fix the maximum length to 100 words.
# pad documents to a max length of n words
max_length = 100
padded_reviews = pad_sequences(encoded_reviews, maxlen=max_length, padding='post')
print(padded_reviews)
```

```python id="xKrq1eXG1AdB"
# We have completed our pre-processing, it is now time to build the neural network based classifier. We start by splitting the reviews into training and test set.
X_train, X_test, y_train, y_test = train_test_split(padded_reviews,Rating,test_size=0.3, random_state=0)
```

<!-- #region id="CBUQuT4G1CCd" -->
Now we need to define the basics of model for neural network
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kw1F5g8QzhDY" executionInfo={"status": "ok", "timestamp": 1628410513708, "user_tz": -330, "elapsed": 1781, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aa51c7d8-bbc0-45ce-c420-f5dd3fd3d9d5"
# define the model
model = Sequential()

# Define the embedding matrix dimensions. Each vector is of 8 dimensions and there will be total of vocab_size vectors
# The input length (window) is 100 words so the output from embedding layer will be a conactenated (flattened) vector of 
# 800 dimensions
model.add(Embedding(vocab_size, 16, input_length=max_length))
model.add(Flatten())
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=1, activation='relu'))

# compile the model with stochastic gradient descent and binary cross entropy
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])

# summarize the model
print(model.summary())
```

```python colab={"base_uri": "https://localhost:8080/"} id="AR1ZNTIG1HuL" executionInfo={"status": "ok", "timestamp": 1628410635710, "user_tz": -330, "elapsed": 15384, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="40185b4e-13d5-49c4-929a-7bfc499fd00f"
# Fit the model... there are few docs, so I am trying with batch_size=1, you can delete it for default batch 
#size or change it to a bigger number
model.fit(X_train, y_train, epochs=10, batch_size=30, verbose=1)
```

<!-- #region id="JAm6R4kI1kDf" -->
Now, we shall evaluate our model against the test set that we kep separate earlier.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ptyT08xP1T50" executionInfo={"status": "ok", "timestamp": 1628410666934, "user_tz": -330, "elapsed": 850, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="865a00ea-c763-4504-af67-37d9e1c9fff4"
# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy: %f' % (accuracy*100))
```

<!-- #region id="bopd8Ytd1pP7" -->
Precision and Recall
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7tEDCIkL1oY4" executionInfo={"status": "ok", "timestamp": 1628410697860, "user_tz": -330, "elapsed": 503, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e1e1b91d-42b4-4a49-fb8b-4bc24e77072c"
predictions = model.predict(X_test, batch_size=100, verbose=1)
predictions_bool = np.argmax(predictions, axis=1)

print(classification_report(y_test, predictions_bool))
```
