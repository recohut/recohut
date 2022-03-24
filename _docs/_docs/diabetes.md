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

<!-- #region id="_VHQRn5uihp7" -->
# Building a Simple Classifier in Keras to Detect Diabetes in Patients
<!-- #endregion -->

<!-- #region id="qPxZLdMIheJ_" -->
## Setup
<!-- #endregion -->

```python id="IaUvh0t2a-PQ"
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy
numpy.random.seed(seed)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
```

```python id="1N7_NLdna_9s"
seed = 7
numpy.random.seed(seed)
```

```python colab={"base_uri": "https://localhost:8080/"} id="OoYWcddCaz4z" executionInfo={"status": "ok", "timestamp": 1637683201271, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="90663a38-0df8-4983-bad1-ec683f043563"
dataset = numpy.loadtxt("https://github.com/sparsh-ai/general-ml/raw/S142234/Hand-On%2BKeras%2B-%2BCase%2BStudy%2BPima%2BIndians%2Bdataset/pima.csv", delimiter=",")
dataset.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="8wEdobs2bGmJ" executionInfo={"status": "ok", "timestamp": 1637683201272, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e80dc038-3fa1-414c-ce44-44f4ed7c8e64"
dataset[:10,:]
```

<!-- #region id="J-Hb0D9LhjwB" -->
## Simple Classifier
<!-- #endregion -->

```python id="aeIr6H22bDAy"
X = dataset[:,0:8]
Y = dataset[:,8]
```

```python colab={"base_uri": "https://localhost:8080/"} id="HcAAOsDObODK" executionInfo={"status": "ok", "timestamp": 1637681624228, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="214f2b64-470c-4256-b105-01f3fde676d3"
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="MRUGuJFSbRip" executionInfo={"status": "ok", "timestamp": 1637681693561, "user_tz": -330, "elapsed": 42102, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b0bdb8cd-cc44-46c0-90c7-a903636c7f24"
#collapse-hide
model.fit(X, Y, epochs=150, batch_size=10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="dlGVgNT5bpeM" executionInfo={"status": "ok", "timestamp": 1637681697635, "user_tz": -330, "elapsed": 565, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c5f6d37d-a6bf-4555-a8e8-b608cb8fe0ec"
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

<!-- #region id="uOqagZeWpb-u" -->
## Classifier with Train/Test Split
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aPr-_Svspj5Y" executionInfo={"status": "ok", "timestamp": 1637685373223, "user_tz": -330, "elapsed": 24573, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9ceb0471-7906-46d4-8c74-d9412d76dd39"
#collapse-hide
X = dataset[:,0:8]
Y = dataset[:,8]

# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)
```

<!-- #region id="K2cRrx0yp5sN" -->
## Classifier with K-fold Cross-validation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PSbbFfHSqC3j" executionInfo={"status": "ok", "timestamp": 1637685591758, "user_tz": -330, "elapsed": 159416, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8afb50b5-85e2-4266-d5e3-cd539ade5d0e"
X = dataset[:,0:8]
Y = dataset[:,8]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train, test in kfold.split(X, Y):
  # create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
	# evaluate the model
	scores = model.evaluate(X[test], Y[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
```

<!-- #region id="MrpICfmphqw1" -->
## Training with checkpoints
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ftQUWdgshqur" executionInfo={"status": "ok", "timestamp": 1637683422679, "user_tz": -330, "elapsed": 20313, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cadafae4-d4fc-4d3d-f897-2f338c510c4b"
X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
```

<!-- #region id="RV-EQh42ioFh" -->
## Loading the best model from disk
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1ifAjxgDhqsP" executionInfo={"status": "ok", "timestamp": 1637683484266, "user_tz": -330, "elapsed": 897, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3350e788-85b0-405a-81f6-fad404755ed3"
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.load_weights("weights.best.hdf5")

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Created model and loaded weights from file")

scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

<!-- #region id="ANs2N2QrbxBl" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bZNZkzeubxBn" executionInfo={"status": "ok", "timestamp": 1637681742834, "user_tz": -330, "elapsed": 3309, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f165e6d7-77af-4af4-b827-18e0d18738ae"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="NBMq1V3AbxBn" -->
---
<!-- #endregion -->

<!-- #region id="ebCJvoi9bxBo" -->
**END**
<!-- #endregion -->
