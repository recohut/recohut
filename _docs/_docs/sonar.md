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

<!-- #region id="L2HGUM7gBfNh" -->
# SONAR Signal Classifier
> Building a Classifier to detect SONAR signal on Sonar Dataset in Keras
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4EOZrYjreYYm" executionInfo={"status": "ok", "timestamp": 1637682370267, "user_tz": -330, "elapsed": 612, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="56c6b0c3-82de-48e9-c333-1b6b9a93c8e5"
!wget -q --show-progress https://raw.githubusercontent.com/hargurjeet/MachineLearning/Sonar-Dataset/sonar.all-data.csv
```

```python id="u_IGkb_uee8l"
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

```python id="g-8pE5imegPP"
seed = 5
numpy.random.seed(seed)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 383} id="o0y_uoh0egZr" executionInfo={"status": "ok", "timestamp": 1637682428223, "user_tz": -330, "elapsed": 780, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4764b8de-eaa2-46b4-e9fc-269051b1f958"
dataframe = read_csv("sonar.all-data.csv", header=None)
dataframe.head(10)
```

```python id="38y_T759eoQC"
dataset = dataframe.values

X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
```

```python id="uTuK4BOHekx3"
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
```

```python id="qd5ZtmvSeu5o"
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

```python colab={"base_uri": "https://localhost:8080/"} id="x12IR5LEewcl" executionInfo={"status": "ok", "timestamp": 1637682517605, "user_tz": -330, "elapsed": 30, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6cfc7903-8d43-40e6-bab4-d8d51aad8abe"
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

```python colab={"base_uri": "https://localhost:8080/"} id="s-pAilwDezKl" executionInfo={"status": "ok", "timestamp": 1637682632633, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9511859e-163e-4102-b0d5-85e1e54e56fc"
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

```python id="KmkFxeUAfYbn"
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=60, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

```python colab={"base_uri": "https://localhost:8080/"} id="JuitfxXUfZFy" executionInfo={"status": "ok", "timestamp": 1637682702727, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="edc259f1-6334-4ebe-f051-b931b96b5d25"
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

```python id="LMI_YqBzfhPg"
def create_larger():
	model = Sequential()
	model.add(Dense(60, input_dim=60, activation='relu'))
	model.add(Dense(60, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

```python colab={"base_uri": "https://localhost:8080/"} id="syIPmZB_fh6D" executionInfo={"status": "ok", "timestamp": 1637682791247, "user_tz": -330, "elapsed": 769, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="31e88369-f796-45f9-f8a5-9d02bd17bdc0"
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```
