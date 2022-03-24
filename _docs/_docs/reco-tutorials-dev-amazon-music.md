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

```python colab={"base_uri": "https://localhost:8080/"} id="zaRyPR2aPsNL" executionInfo={"status": "ok", "timestamp": 1622487537613, "user_tz": -330, "elapsed": 412, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e8cb03ea-e5d0-40c1-b3e1-aa6002e21355"
# !git clone https://github.com/MrRezaeiUofT/Amazon-Product-Recommender.git
# !cd Amazon-Product-Recommender && zip -s- Dataset.zip -O /content/AmazonDataset
# !unzip AmazonDataset.zip
# !mkdir AmazonMusicRatings && cd AmazonMusicRatings && \
# cp /content/train.json/train.json . && \
# cp /content/test.json/test.json . && \
# cp /content/rating_pairs.csv .
# %cd AmazonMusicRatings
```

```python id="r-_QIw7Fq0hR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1622493335680, "user_tz": -330, "elapsed": 436, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ad5f1cee-d9c4-4671-e973-9a317176edbe"
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import tensorflow.keras.backend as K
import tensorflow as tf
import string
import snowballstemmer
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.manifold import TSNE
import numpy as np
from collections import defaultdict
import json
from collections import defaultdict
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Dropout,Flatten,Dense,TimeDistributed,Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
np.random.seed(0)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Dropout,Flatten,Dense,TimeDistributed,Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
np.random.seed(0)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Dropout,Flatten,Dense,TimeDistributed,Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
np.random.seed(0)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
```

```python id="Tws-ZYtnq3ur"
MAX_WORD_TO_USE = 1000 # how many words to use in training
MAX_WORD_TO_USE_other=100
MAX_LEN = 100 # number of time-steps.
```

```python id="ifpmYdPbq8cG"
def Text_cleaner(Data):
    HoleText = []
    separator = ' '
    for i in range(Data.shape[0]):
        HoleText.append(separator.join(text_to_word_sequence(str(Data[i]))))
    x_c = HoleText
    return x_c
```

```python id="Fw6wB4_7rAeH"
def Text_length(Data):
    HoleText = []
    for i in range(Data.shape[0]):
        HoleText.append(text_to_word_sequence(str(Data[i])))
    x_c = HoleText
    x_c_l=[len(x_c[i]) for i in range(len(x_c))]
    return x_c_l
```

```python id="PYJSp4MorJvy"
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
```

```python id="5ZDPDvmirMwu"
def weighted_mse(weights):
    """
    A weighted version of MSE
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.

        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        # y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = tf.reduce_logsumexp(tf.pow(y_true - y_pred,2) * weights)

        return loss

    return loss
```

```python id="XM1QF8Y3qSPt"
def clean_text(text):
    ## Remove puncuation
    text = text.translate(string.punctuation)

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Stemming
    text = text.split()
    stemmer = nltk.stem.SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text
```

```python id="EAesjcp3rP0r" executionInfo={"status": "ok", "timestamp": 1622487745212, "user_tz": -330, "elapsed": 644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# allRatings = []
# userRatings = defaultdict(list)

# with open('train.json', 'r') as train_file:
#     for row in train_file:
#         data = json.loads(row)
#         r = float(data['overall'])
#         allRatings.append(r)
#         userRatings[data['reviewerID']].append(r)

# globalAverage = sum(allRatings)/len(allRatings)
# userAverage = {}
# for u in userRatings:
#     userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

# predictions = open('rating_predictions.csv', 'w')
# for l in open('rating_pairs.csv'):
#     if l.startswith('userID'):
#         #header
#         predictions.write(l)
#         continue
#     u,p = l.strip().split('-')
#     if u in userAverage:
#         predictions.write(u + '-' + p + ',' + str(userAverage[u]) + '\n')
#     else:
#         predictions.write(u + '-' + p + ',' + str(globalAverage) + '\n')
# predictions.close()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="fv0Wk5Qb5Zbe" executionInfo={"status": "ok", "timestamp": 1622492070125, "user_tz": -330, "elapsed": 2079, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="306a4e89-a0a1-4b09-f82c-2bea5b222d19"
data = []
bad_records = 0
with open('train.json') as f:
    for line in f:
      try:
        data.append(json.loads(line))
      except:
        bad_records+=1
        pass
print('Bad records skipped: {}'.format(bad_records))

df = pd.DataFrame.from_records(data).reset_index()
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 408} id="VvAmdI0GEDxB" executionInfo={"status": "ok", "timestamp": 1622492099068, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e2011e93-e6a9-43ad-f71f-d2a286769dd0"
rate_pair_df = pd.read_json('test.json', lines=True)
rate_pair_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="FjIi3RRYEOes" executionInfo={"status": "ok", "timestamp": 1622492131060, "user_tz": -330, "elapsed": 395, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ab2e2dc4-6b94-4810-aa42-8f5d6d09791b"
rate_pair_id = pd.read_csv('rating_pairs.csv')
rate_pair_id.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="8hkH3HuIEUsQ" executionInfo={"status": "ok", "timestamp": 1622492158622, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8522039d-8dea-4dcf-d6bf-280bd87fa264"
compensate_number = df.overall.value_counts()[5]
compensate_number
```

```python id="C9Ewm3-lzxNz" executionInfo={"status": "ok", "timestamp": 1622492164479, "user_tz": -330, "elapsed": 415, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# MaxWordLength=50
# Batch_Size = 20000
# Dp_rate=0.2
# Epochs=200
# vocabulary_size = 2000
```

```python id="RkrDMxFLF2SI" executionInfo={"status": "ok", "timestamp": 1622494855997, "user_tz": -330, "elapsed": 658, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def preprocess(train, test, vocabulary_size):

  train.reset_index(drop=True, inplace=True)
  test.reset_index(drop=True, inplace=True)

  # combine summary and review text columns
  text = []
  for i in range(train.shape[0]):
    text.append(clean_text(str(train['summary'][i]) + str(train['reviewText'][i])))
  for i in range(test.shape[0]):
    text.append(clean_text(str(test['summary'][i]) + str(test['reviewText'][i])))

  # text tokenization
  tokenizer = Tokenizer(num_words=vocabulary_size)
  tokenizer.fit_on_texts(text)
  F_text = tokenizer.texts_to_matrix(text, mode='freq')

  F_text = F_text[:train.shape[0],:]
  F_text_te = F_text[train.shape[0]:,:]

  # time normalization
  F_time = (train.unixReviewTime.values - train.unixReviewTime.min())/(train.unixReviewTime.max() - train.unixReviewTime.min())
  F_time = F_time.reshape([-1,1])

  # dummy encoding of category column
  F_cat = pd.get_dummies(train.category).values

  newD = pd.DataFrame(np.concatenate([train['itemID'].values, test['itemID'].values], axis=0), columns=['itemID'])
  newD['reviewerID'] = np.concatenate([train['reviewerID'].values, test['reviewerID'].values], axis=0)
  newD['itemID'] = newD['itemID'].groupby(newD['itemID']).transform('count')
  newD['reviewerID'] = newD['reviewerID'].groupby(newD['reviewerID']).transform('count')

  RID_temp = pd.get_dummies(newD['reviewerID']).values
  F_newD_reviewerID = RID_temp[:train.shape[0],:]
  F_newD_reviewerID_te = RID_temp[train.shape[0]:,:]

  IID_temp = pd.get_dummies(newD['itemID']).values
  F_newD_itemID = IID_temp[:train.shape[0],:]
  F_newD_itemID_te = IID_temp[train.shape[0]:,:]

  vocab_size_RID=RID_temp.max()+1
  vocab_size_IID=IID_temp.max()+1

  return F_text, F_time, F_cat, F_newD_reviewerID, F_newD_itemID, F_newD_itemID_te, vocab_size_RID, vocab_size_IID
```

```python id="Sg4aFxqn0QBX" executionInfo={"status": "ok", "timestamp": 1622494342961, "user_tz": -330, "elapsed": 2287, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
temp_train = df.sample(1000).reset_index(drop=True)
temp_test = rate_pair_df.sample(100).reset_index(drop=True)

F_text, F_time, F_cat, F_newD_reviewerID, F_newD_itemID, _, _, _ = preprocess(temp_train, temp_test, vocabulary_size)

y = temp_train.overall.to_numpy()
X = np.concatenate([F_text, F_time, F_cat, F_newD_reviewerID, F_newD_itemID], axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1vw5WjcFKYhc" executionInfo={"status": "ok", "timestamp": 1622494344640, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c75d569e-438b-4ade-a9e0-34759a9b067f"
########### naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)
print('Naive_bayesian RMS=%f'%(metrics.mean_squared_error(y_test, nb_preds)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="TUtszP_JKbWx" executionInfo={"status": "ok", "timestamp": 1622494354692, "user_tz": -330, "elapsed": 4467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3ce3a0c0-45ab-447d-e905-ed160e6b6248"
########### SVM
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5)
# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X_train, y_train)
rl_preds=logreg.predict(X_test)
print('Logistic Regression RMS=%f'%(metrics.mean_squared_error(y_test,rl_preds)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="bU0sjMisOyoh" executionInfo={"status": "ok", "timestamp": 1622494904340, "user_tz": -330, "elapsed": 539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ca1a85f9-343e-4f46-fe7b-70eebc676ea4"
# x = preprocess(temp_train, temp_test, vocabulary_size)

```

```python colab={"base_uri": "https://localhost:8080/"} id="GZ8geHQ3LawE" outputId="1cc54bcd-d769-48c4-f779-78903c0752d7"
MaxWordLength=50
Batch_Size = 10000
Dp_rate=0.5
Epochs=10
vocabulary_size = 5000

temp_train = df.sample(1000).reset_index(drop=True)
temp_test = rate_pair_df.sample(100).reset_index(drop=True)

F_text, F_time, F_cat, F_newD_reviewerID, F_newD_itemID, F_newD_itemID_te, vocab_size_RID, vocab_size_IID = preprocess(temp_train, temp_test, vocabulary_size)

''' Model Selection'''
x_text = Input(shape=(F_text.shape[1]),name='x_text')
x_time = Input(shape=(F_time.shape[1]),name='x_time')
x_cat = Input(shape=(F_cat.shape[1]),name='x_cat')
x_RID = Input(shape=(F_newD_reviewerID.shape[1]),name='x_RID')
x_IID = Input(shape=(F_newD_itemID_te.shape[1]),name='x_IID')

H_t=tf.keras.layers.Embedding(vocabulary_size, 100)(x_text)
H_t=LSTM(100, dropout=Dp_rate, recurrent_dropout=Dp_rate)(H_t)

H_RID=tf.keras.layers.Embedding(vocab_size_RID, 20)(x_RID)
H_RID=tf.reshape(H_RID, (-1,20))

H_IID=tf.keras.layers.Embedding(vocab_size_IID, 20)(x_IID)
H_IID=tf.reshape(H_IID, (-1,20))

H_c=Dense(5,activation='relu')(x_cat)
H_c=Dropout(Dp_rate)(H_c)

H=tf.concat([H_t,H_RID,H_IID,x_cat,x_time],axis=-1)

# Final predictions and model.
prediction = Dense(1, activation='sigmoid')(H)
model = Model(inputs=[x_text,x_time,x_cat,x_RID,x_IID], outputs= prediction)
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
model.compile(loss=tf.keras.losses.MSE,
              optimizer=optimizer,metrics=['mse'])
model.summary()

''' Train mode'''
history=model.fit([F_text,F_time,F_cat,F_newD_reviewerID,F_newD_itemID], y/5,
          batch_size=Batch_Size,
          epochs=Epochs,
          verbose=1,
          validation_split=.3, shuffle=True)
plt.figure(figsize=(12,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='lower right')
plt.show()
```

```python id="BW2-8ZmozN-z"
''' Model Selection'''
x_text = Input(shape=(F_text.shape[1]),name='x_text')
x_time = Input(shape=(F_time.shape[1]),name='x_time')
x_cat = Input(shape=(F_cat.shape[1]),name='x_cat')
x_RID = Input(shape=(F_newD_reviewerID.shape[1]),name='x_RID')
x_IID = Input(shape=(F_newD_itemID_te.shape[1]),name='x_IID')

H_t=tf.keras.layers.Embedding(vocabulary_size, 100)(x_text)
H_t=LSTM(100, dropout=Dp_rate, recurrent_dropout=Dp_rate)(H_t)


H_RID=tf.keras.layers.Embedding(vocab_size_RID, 20)(x_RID)
H_RID=tf.reshape(H_RID, (-1,20))


H_IID=tf.keras.layers.Embedding(vocab_size_IID, 20)(x_IID)
H_IID=tf.reshape(H_IID, (-1,20))

H_c=Dense(5,activation='relu')(x_cat)
H_c=Dropout(Dp_rate)(H_c)

H=tf.concat([H_t,H_RID,H_IID,x_cat,x_time],axis=-1)

# Final predictions and model.
prediction = Dense(1, activation='sigmoid')(H)
model = Model(inputs=[x_text,x_time,x_cat,x_RID,x_IID], outputs= prediction)
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
model.compile(loss=tf.keras.losses.MSE,
              optimizer=optimizer,metrics=['mse'])
model.summary()

''' Train mode'''
history=model.fit([F_text,F_time,F_cat,F_newD_reviewerID,F_newD_itemID], y/5,
          batch_size=Batch_Size,
          epochs=Epochs,
          verbose=1,
          validation_split=.3, shuffle=True)
plt.figure(figsize=(12,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='lower right')
plt.show()
```

```python id="7BmOi1ihzN8j"
Text=[]
for i in range(Data.shape[0]):
    # print(i)
    Text.append(clean_text(str(Data['summary'][i]) + str(Data['reviewText'][i]) ))
for i in range(RatePairData.shape[0]):

    Text.append(clean_text(str(RatePairData['summary'][i]) + str(RatePairData['reviewText'][i]) ))

tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(Text)
F_textD = tokenizer.texts_to_matrix(Text, mode='freq')
# F_textD = pad_sequences(sequences, maxlen=MaxWordLength)
F_text=F_textD[:Data.shape[0],:]
F_text_te=F_textD[Data.shape[0]:,:]
'''P2--> Normalize unixReviewTime'''
F_time=( Data.unixReviewTime.values  - Data.unixReviewTime.min())/(Data.unixReviewTime.max() - Data.unixReviewTime.min())
F_time=F_time.reshape([-1,1])
'''P3--> one-hot encoding of categories'''
F_cat=pd.get_dummies(Data.category).values
'''P4 --> for later'''
''' Part 5 --> add popularity of an product as feature. I used histigram method'''
newD=pd.DataFrame(np.concatenate([Data['itemID'].values,RatePairData['itemID'].values],axis=0),columns=['itemID'])
newD['reviewerID']=np.concatenate([Data['reviewerID'].values,RatePairData['reviewerID'].values],axis=0)
newD['itemID']=newD['itemID'].groupby(newD['itemID']).transform('count')
newD['itemID']=(newD['itemID'].values-newD['itemID'].min())/(newD['itemID'].max()-newD['itemID'].min())

newD['reviewerID']=newD['reviewerID'].groupby(newD['reviewerID']).transform('count')
newD['reviewerID']=(newD['reviewerID'].values-newD['reviewerID'].min())/(newD['reviewerID'].max()-newD['itemID'].min())
newD=newD.to_numpy()
F_newD=newD[:Data.shape[0]]
F_newD_te=newD[Data.shape[0]:]

'''test-train split '''
y = Data['overall']
y=Data.overall.to_numpy()

''' Model Selection'''
x_text = Input(shape=(F_text.shape[1]),name='x_text')
x_time = Input(shape=(F_time.shape[1]),name='x_time')
x_cat = Input(shape=(F_cat.shape[1]),name='x_cat')
x_newD = Input(shape=(F_newD.shape[1]),name='x_newD')


H=Dense(MaxWordLength,activation='relu')(x_text)
H=Dropout(Dp_rate)(H)
H=Dense(MaxWordLength,activation='relu')(H)
H=Dropout(Dp_rate)(H)
H=tf.concat([H,x_time,x_cat,x_newD],axis=-1)
H=Dense(MaxWordLength, activation='relu')(H)
H=Dense(MaxWordLength, activation='relu')(H)
# Final predictions and model.
prediction = Dense(1, activation='sigmoid')(H)
# kernel_initializer='ones',
#     kernel_regularizer=tf.keras.regularizers.L1(0.01),
#     activity_regularizer=tf.keras.regularizers.L2(0.01)
model = Model(inputs=[x_text,x_time,x_cat,x_newD], outputs= prediction)
model.compile(loss=tf.keras.losses.MSE,
              optimizer='adam')
model.summary()

''' Train modek'''
history=model.fit([F_text,F_time,F_cat,F_newD], y/5,
          batch_size=Batch_Size,
          epochs=Epochs,
          verbose=1,
          validation_split=0.0,shuffle=True)
plt.figure(figsize=(12,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='lower right')
plt.show()
##################### prediction
'''P2--> Normalize unixReviewTime'''
F_time_te=( RatePairData.unixReviewTime.values  - Data.unixReviewTime.min())/(Data.unixReviewTime.max() - Data.unixReviewTime.min())
F_time_te=F_time_te.reshape([-1,1])
'''P3--> one-hot encoding of categories'''
F_cat_te=pd.get_dummies(RatePairData.category).values

y_pred = model([F_text_te,F_time_te,F_cat_te,F_newD_te])
RatePairID.prediction=y_pred.numpy()*5


RatePairID.to_csv('rating_predictions.csv',index=False)
```

```python id="AM61Lsn2zN5z"

```
