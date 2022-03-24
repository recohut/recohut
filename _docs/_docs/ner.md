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

# NER

```python colab={"base_uri": "https://localhost:8080/", "height": 343} colab_type="code" executionInfo={"elapsed": 9561, "status": "ok", "timestamp": 1587735551418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="npGOxsIriHlw" outputId="b6ed949f-eaf4-46b6-e1d5-1e42f708ba08"
!wget -x --load-cookies cookies.txt -O data.zip "https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/download/x3fQmLPgJrYvJpSDyApN%2Fversions%2F3dilmDFPVvpZOIR0qKZQ%2Ffiles%2Fner_dataset.csv?datasetVersionNumber=4"
!unzip data.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} colab_type="code" executionInfo={"elapsed": 1594, "status": "ok", "timestamp": 1587735585581, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="xIDBwykYj0eJ" outputId="de3f1252-537b-4b71-96c0-c04bb66a1327"
import pandas as pd
import numpy as np

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
data.head(10)
```

```python colab={} colab_type="code" id="chQWqYoW2a3C"
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
    
    def get_next(self):
        try:
            s = self.data[self.data["Sentence #"] == "Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s["Word"].values.tolist(), s["POS"].values.tolist(), s["Tag"].values.tolist()    
        except:
            self.empty = True
            return None, None, None

```

```python colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" executionInfo={"elapsed": 1653, "status": "ok", "timestamp": 1587735647037, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="r4Ce3nlkavdG" outputId="10ad34c3-ab18-4ef4-dadc-abdc993b7569"
getter = SentenceGetter(data)
sent, pos, tag = getter.get_next()
print(sent); print(pos); print(tag)
```

```python colab={} colab_type="code" id="h7hYWo7da0x_"
from sklearn.base import BaseEstimator, TransformerMixin


class MemoryTagger(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y):
        '''
        Expects a list of words as X and a list of tags as y.
        '''
        voc = {}
        self.tags = []
        for x, t in zip(X, y):
            if t not in self.tags:
                self.tags.append(t)
            if x in voc:
                if t in voc[x]:
                    voc[x][t] += 1
                else:
                    voc[x][t] = 1
            else:
                voc[x] = {t: 1}
        self.memory = {}
        for k, d in voc.items():
            self.memory[k] = max(d, key=d.get)
    
    def predict(self, X, y=None):
        '''
        Predict the the tag from memory. If word is unknown, predict 'O'.
        '''
        return [self.memory.get(x, 'O') for x in X]

```

```python colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" executionInfo={"elapsed": 1428, "status": "ok", "timestamp": 1587735729689, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="IvqcvCaYa5-Y" outputId="e78ba342-b43e-4192-c9d0-6ebae44f31aa"
tagger = MemoryTagger()
tagger.fit(sent, tag)
print(tagger.predict(sent))
print(['O', 'B-geo', 'B-gpe'])
```

```python colab={} colab_type="code" id="DugoJh_ta-XE"
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

words = data["Word"].values.tolist()
tags = data["Tag"].values.tolist()

pred = cross_val_predict(estimator=MemoryTagger(), X=words, y=tags, cv=5)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 425} colab_type="code" executionInfo={"elapsed": 11005, "status": "ok", "timestamp": 1587735874386, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="XRnixxrWbPvK" outputId="6337929b-4052-4b4d-9397-918fabedec6b"
report = classification_report(y_pred=pred, y_true=tags)
print(report)
```

<!-- #region colab_type="text" id="-hkVS_BGbrZs" -->
## Frequency + Random Forest
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 479} colab_type="code" executionInfo={"elapsed": 45621, "status": "ok", "timestamp": 1587735948420, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="6cG1Q_fjbZq3" outputId="88fa418d-3556-4091-b4c5-f5f355f8ef4f"
from sklearn.ensemble import RandomForestClassifier

def feature_map(word):
    '''Simple feature map.'''
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word), word.isdigit(), word.isalpha()])

words = [feature_map(w) for w in data["Word"].values.tolist()]

pred = cross_val_predict(RandomForestClassifier(n_estimators=20),
                         X=words, y=tags, cv=5)

report = classification_report(y_pred=pred, y_true=tags)
print(report)
```

```python colab={} colab_type="code" id="xqmpKbhXbzoD"
from sklearn.preprocessing import LabelEncoder

class FeatureTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.memory_tagger = MemoryTagger()
        self.tag_encoder = LabelEncoder()
        self.pos_encoder = LabelEncoder()
        
    def fit(self, X, y):
        words = X["Word"].values.tolist()
        self.pos = X["POS"].values.tolist()
        tags = X["Tag"].values.tolist()
        self.memory_tagger.fit(words, tags)
        self.tag_encoder.fit(tags)
        self.pos_encoder.fit(self.pos)
        return self
    
    def transform(self, X, y=None):
        def pos_default(p):
            if p in self.pos:
                return self.pos_encoder.transform([p])[0]
            else:
                return -1
        
        pos = X["POS"].values.tolist()
        words = X["Word"].values.tolist()
        out = []
        for i in range(len(words)):
            w = words[i]
            p = pos[i]
            if i < len(words) - 1:
                wp = self.tag_encoder.transform(self.memory_tagger.predict([words[i+1]]))[0]
                posp = pos_default(pos[i+1])
            else:
                wp = self.tag_encoder.transform(['O'])[0]
                posp = pos_default(".")
            if i > 0:
                if words[i-1] != ".":
                    wm = self.tag_encoder.transform(self.memory_tagger.predict([words[i-1]]))[0]
                    posm = pos_default(pos[i-1])
                else:
                    wm = self.tag_encoder.transform(['O'])[0]
                    posm = pos_default(".")
            else:
                posm = pos_default(".")
                wm = self.tag_encoder.transform(['O'])[0]
            out.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
                                 self.tag_encoder.transform(self.memory_tagger.predict([w]))[0],
                                 pos_default(p), wp, wm, posp, posm]))
        return out
```

```python colab={} colab_type="code" id="1cdzv4lkb8G-"
from sklearn.pipeline import Pipeline

pred = cross_val_predict(Pipeline([("feature_map", FeatureTransformer()), 
                                   ("clf", RandomForestClassifier(n_estimators=20, n_jobs=3))]),
                         X=data, y=tags, cv=5)

report = classification_report(y_pred=pred, y_true=tags)
print(report)
```

<!-- #region colab_type="text" id="_1y6YzHBcP5K" -->
## CRF
<!-- #endregion -->

```python colab={} colab_type="code" id="aS_g5835b_aJ"
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

```

```python colab={} colab_type="code" id="8FL20tnWck4r"
getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)
```

```python colab={} colab_type="code" id="T2Tzc3T5cnYa"
sentences = getter.sentences
```

```python colab={} colab_type="code" id="hr5ArvkFcpF4"
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

```

```python colab={} colab_type="code" id="tN5C1mLYcs-Z"
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]
```

```python colab={} colab_type="code" id="TMS-16ICcunA"
from sklearn_crfsuite import CRF

crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)
```

```python colab={} colab_type="code" id="aWbyE1kuc1L1"
from sklearn.cross_validation import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)

report = flat_classification_report(y_pred=pred, y_true=y)
print(report)
```

```python colab={} colab_type="code" id="6wm2VSZGc1d_"
crf.fit(X, y)
```

```python colab={} colab_type="code" id="uCDT5c7odJr3"
import eli5
eli5.show_weights(crf, top=30)
```

```python colab={} colab_type="code" id="4SWDKt7vdrDx"
crf = CRF(algorithm='lbfgs',
c1=10,
c2=0.1,
max_iterations=100,
all_possible_transitions=False)

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)

report = flat_classification_report(y_pred=pred, y_true=y)
print(report)
```

```python colab={} colab_type="code" id="kJMmZkTmdvtr"
crf.fit(X, y)
```

```python colab={} colab_type="code" id="jSKbXtW_dzcL"
eli5.show_weights(crf, top=30)
```
