---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: papermill-mlflow
    language: python
    name: papermill-mlflow
---

# SNP Data Analysis and Modeling

```python id="-d8Cecspbj5u"
# Tag this cell as 'parameters'
INDEX = 'SNP'
```

```python id="YALQnlmKbOWe"
%load_ext autoreload
%autoreload 2
```

```python id="hPLPoTPZbOWu"
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from joblib import load

from config import EXPERIMENT, RUN_NAME, ARTIFACT_DIR
from logger import logger
from mlflow_utils import evaluate_binary, log_mlflow
```

```python id="WH6BGPapbOWx"
# Wrap all this is a parameter dict
run_params = {'experiment': EXPERIMENT,
              'iteration': RUN_NAME,
              'index': INDEX,
              'artifact_dir': ARTIFACT_DIR}
```

<!-- #region id="R3eB_CPVbOWy" -->
### Import data
<!-- #endregion -->

```python id="R25OeWiObOWy"
df = pd.read_csv('../datadir/{}.csv'.format(INDEX))
```

<!-- #region id="HRtL-_9obOWz" -->
### Data prep
<!-- #endregion -->

```python id="7QgSfx8ZbOW0"
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
```

```python id="03HRQ1k6bOW1"
df.sort_values(by='date', ascending=True, inplace=True)  # Have to sort ascending for rolling to work correctly
```

<!-- #region id="g2ww0zFRbOW2" -->
### Create basic features
<!-- #endregion -->

```python id="li-zdf8_bOW3"
df = df[['date', 'open', 'adj_close']].copy()
```

```python id="9u1ec5qebOW3"
df['prev'] = df['adj_close'].shift(1)
```

```python id="54HVPN8sbOW4"
df['sma2'] = df['prev'].rolling(window=2).mean()
df['sma5'] = df['prev'].rolling(window=5).mean()
df['sma10'] = df['prev'].rolling(window=10).mean()
df['sma20'] = df['prev'].rolling(window=20).mean()
```

```python id="lczhOosRbOW4"
assert round(df['prev'].tail(5).mean(), 4) == round(df['sma5'].tail(1).item(), 4), 'Expected sma5 to be same as mean of past 5 items'
assert round(df['prev'].tail(10).mean(), 4) == round(df['sma10'].tail(1).item(), 4), 'Expected sma10 to be same as mean of past 10 items'
assert round(df['prev'].tail(20).mean(), 4) == round(df['sma20'].tail(1).item(), 4), 'Expected sma20 to be same as mean of past 20 items'
```

```python id="E0fRxMSVbOW5"
df['ema12'] = df['prev'].ewm(span=12, min_periods=12, adjust=False).mean()
df['ema26'] = df['prev'].ewm(span=26, min_periods=26, adjust=False).mean()
```

```python id="5pVSOfkhbOW7"
def difference(prev_price, moving_average):
    return (prev_price - moving_average) / prev_price
```

```python id="txKZy3WzbOW8"
for col in ['sma2', 'sma5', 'sma10', 'sma20', 'ema12', 'ema26']:
    df['{}_diff'.format(col)] = difference(df['prev'], df[col])
```

```python id="XjXM8kQTbOW-"
df.dropna(inplace=True)
```

```python id="toGWfaK2bOW_" outputId="d3014976-71cf-4df3-da51-3f38337b2597"
df.shape
```

```python id="JbjavDdYbOXC"
df.drop(columns=['date', 'open'], inplace=True)
```

<!-- #region id="_eBtQcNqbOXD" -->
### Create label
<!-- #endregion -->

```python id="niWqdLlBbOXD"
df['label'] = (df['adj_close'] > df['prev']).astype(int)
```

<!-- #region id="dakt7Z8PbOXE" -->
### Train-val split
<!-- #endregion -->

```python id="WJJe2DPUbOXE"
# Keep most recent 20 days as validation set
```

```python id="9bTso6zabOXE" outputId="0bd709c1-cf5c-45ef-8360-b2ea976d9eee"
validation_days = 20
train = df.iloc[:-validation_days].copy()
val = df.iloc[-validation_days:].copy()
logger.info('No. of days - Train: {}, Val: {}'.format(train.shape[0], val.shape[0]))
```

<!-- #region id="0M28UW_SbOXF" -->
### Basic visualizations
<!-- #endregion -->

```python id="K_PsQhtHbOXF" outputId="99923b52-fae4-438c-94c5-ebf4758a2f26"
plt.figure(figsize=(15, 5))
plt.grid()
plt.plot(train['adj_close'], label='price')
plt.plot(train['sma20'], label='simple moving average 20')
plt.plot(train['ema12'], label='exponential moving average 12')
plt.plot(train['ema26'], label='exponential moving average 26')
plt.legend()
```

<!-- #region id="pceDdFxLbOXG" -->
### Feature transformations
<!-- #endregion -->

```python id="6feP7kYCbOXG"
scaler = StandardScaler()
```

```python id="Puyms6gwbOXH"
COLS_TO_SCALE = ['prev', 'sma2', 'sma5', 'sma10', 'sma20', 'ema12', 'ema26']
```

```python id="DNw-r02UbOXH" outputId="d2164916-25a7-4d6d-d071-96b1ab7a114f"
scaler.fit(train[COLS_TO_SCALE])
```

```python id="95nRVQerbOXI"
train[COLS_TO_SCALE] = scaler.transform(train[COLS_TO_SCALE])
val[COLS_TO_SCALE] = scaler.transform(val[COLS_TO_SCALE])
```

<!-- #region id="1o7xXj4wbOXI" -->
### Linear regression coefficients
<!-- #endregion -->

```python id="wbT5YzvZbOXJ" outputId="13a747ea-02ce-4f2f-e1f6-4e10981f234e"
x_variables = '+'.join(list(train.columns[1:-1]))
logger.info('x variables: {}'.format(x_variables))
```

```python id="X_MgXrcvbOXJ"
results = smf.ols('adj_close ~ {}'.format(x_variables), data=train).fit()
```

```python id="OBrnc-8ZbOXK" outputId="56396608-2313-4a3c-de0d-65115fa811a1"
print(results.summary())
```

<!-- #region id="tL_YTrSqbOXK" -->
### Drop label col
<!-- #endregion -->

```python id="gSAtTJGxbOXL"
train.drop(columns=['adj_close'], inplace=True)
val.drop(columns=['adj_close'], inplace=True)
```

<!-- #region id="thrWOw8qbOXL" -->
### Logistic Regression Coefficients
<!-- #endregion -->

```python id="UP8ijlOEbOXM"
y = train['label'].values
X = train.drop(columns=['label'])
```

```python id="QTOGPhFibOXM"
X = add_constant(X)
```

```python id="thQgfowzbOXN"
logit = sm.Logit(y, X)
```

```python id="udLugFfabOXN" outputId="35fa2183-2184-4007-c55a-55f332da0000"
result = logit.fit()
print(result.summary())
```

<!-- #region id="UeEud8phbOXO" -->
### Train some basic models
<!-- #endregion -->

```python id="8HHpdiksbOXO"
y_train = train['label'].values
X_train = train.drop(columns='label').values

y_val = val['label'].values
X_val = val.drop(columns='label').values
```

```python id="u-pmKm23bOXP" outputId="bd6b7ba1-9a8d-43e4-caa3-078c57f2ffcd"
# Logistic regression
model_name = 'Logistic Regression'
model = LogisticRegression(fit_intercept=False).fit(X_train, y_train)
pred = model.predict_proba(X_val)[:, 1]
log_mlflow(run_params, model, model_name, y_val, pred)
```

```python id="qmRDr8F4bOXP" outputId="8681fb1d-2e25-4b73-f650-033edcbd9d2d"
# Support vector classifier
model_name = 'Support Vector Classifier'
model = SVC(gamma=2, C=1, probability=True).fit(X_train, y_train)
pred = model.predict_proba(X_val)[:, 1]
log_mlflow(run_params, model, model_name, y_val, pred)
```

```python id="N3vvrIBabOXQ" outputId="83fa9bc8-ad69-4bd6-8f5c-9aac59fb56b1"
# K-neighbours
model_name = 'K-nearest Neighbours'
model = KNeighborsClassifier(5).fit(X_train, y_train)
pred = model.predict_proba(X_val)[:, 1]
log_mlflow(run_params, model, model_name, y_val, pred)
```

```python id="RgkR3wR3bOXR" outputId="7f0276c8-976f-4044-ae72-b390e7a2b0ce"
# Gradient Boosting
model_name = 'Gradient Boosting Machine'
model = GradientBoostingClassifier().fit(X_train, y_train)
pred = model.predict_proba(X_val)[:, 1]
log_mlflow(run_params, model, model_name, y_val, pred)
```

```python id="XIif9RqpbOXR" outputId="11ded34f-86f2-41c4-c875-e1faefc82ce5"
# Bagged trees
model_name = 'Extra Trees'
model = ExtraTreesClassifier().fit(X_train, y_train)
pred = model.predict_proba(X_val)[:, 1]
log_mlflow(run_params, model, model_name, y_val, pred)
```

<!-- #region id="yazk6nv4bOXS" -->
### Test loading model
<!-- #endregion -->

```python id="1ohER0thbOXS"
model_name = 'Support Vector Classifier'
check_model = load('{}/models/{}.pickle'.format(ARTIFACT_DIR, model_name))
```

```python id="-UnneOvobOXT" outputId="04fcec13-b2a8-4dd4-b4f9-55ebb332bbe0"
check_model
```

```python id="8QqfhgHCbOXT"
pred = check_model.predict_proba(X_val)[:, 1]
```

```python id="T7VOVqBubOXU" outputId="73ee5f15-6152-43a0-db58-ab4fd33a6d48"
evaluate_binary(y_val, pred)
```

```python id="qa-2Cs2cbOXU"

```
