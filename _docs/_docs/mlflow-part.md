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

# MLFlow Part 3

<!-- #region id="xnt87vd1A622" -->
## Environment setup
<!-- #endregion -->

```python executionInfo={"elapsed": 685, "status": "ok", "timestamp": 1630410173387, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="U57Wh7-QwikM"
import os
project_name = "reco-tut-de"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="x8AO_wohwn9G"
if not os.path.exists(project_path):
    !pip install -U -q dvc dvc[gdrive]
    !pip install -q mlflow
    !apt-get install tree
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

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 503, "status": "ok", "timestamp": 1630412816289, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="OrqXufqUwn9J" outputId="ec90e754-e04a-430c-a987-285a3f2bbfaa"
!git status
```

```python id="BPVM-5NKwn9K"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14796, "status": "ok", "timestamp": 1630412856937, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="jYnf6l5vEGtA" outputId="2606a38b-83cb-47f4-c7b1-2da7f89f90ef"
!dvc commit && dvc push
```

```python id="c9YrvZGcq_Ou"
%reload_ext autoreload
%autoreload 2
```

```python id="bwaUt8ktjPru"
!make setup
```

<!-- #region id="JsH02Sj3CvKU" -->
### Pull specific data file
<!-- #endregion -->

```python id="g_o2ApQ2squS"
!dvc pull ./data/silver/stockpred/train.csv.dvc
```

<!-- #region id="2fZNPDKICx7t" -->
### Reinitiate old project - ```stockpred_comparisons```
<!-- #endregion -->

```python id="6QvXYI0uEeyD"
!cd /content/reco-tut-de && dvc pull -q ./src/mlflow/stockpred_comparisons/mlruns.dvc
```

```python colab={"base_uri": "https://localhost:8080/", "height": 51} executionInfo={"elapsed": 12167, "status": "ok", "timestamp": 1630410761106, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Jcgpt1kEDBm7" outputId="0d7fb0f9-d6a9-477e-c66f-54d32fde7e86"
from src.mlflow.utils import MLFlow

stockpred = MLFlow()

stockpred.create_project(name='stockpred_comparisons',
                         basepath='/content/reco-tut-de/src/mlflow',
                         entryfile='train.py')

stockpred.get_ui()
```

<!-- #region id="SonEtDZTAb8_" -->
## Load model as a PyFuncModels
<!-- #endregion -->

```python executionInfo={"elapsed": 405, "status": "ok", "timestamp": 1630411403602, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="uMefhaXeFsEQ"
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1630411405276, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="UJaH2YR-ESLK" outputId="49453cbb-379e-4161-f43c-5daa4ad22ec1"
logged_model = './mlruns/1/f1ccd1a06c3d4eec863dc1816f588b40/artifacts/model'

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Load Data
pandas_df = pd.read_csv(os.path.join(project_path,'data/silver/stockpred/train.csv'))
X = pandas_df.iloc[:,:-1]
Y = pandas_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=4284, stratify=Y)

# Predict on a Pandas DataFrame
loaded_model.predict(pd.DataFrame(X_test))
```

<!-- #region id="DFxTxRWo3EQf" -->
## Hyperparameter Tuning
<!-- #endregion -->

```python id="TA_QU6ZY36Vu"
!pip install -q hyperopt
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1654, "status": "ok", "timestamp": 1630412314783, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="VE_m8OG8H6RF" outputId="21c0d09b-d0bb-4b52-8867-0d9e1b8530ac"
# Import variables
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas
import mlflow

# Retrieve Data
pandas_df = pandas.read_csv(os.path.join(project_path,'data/silver/stockpred/train.csv'))
X = pandas_df.iloc[:,:-1]
Y = pandas_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=4284, stratify=Y)

# Define objective function
N_FOLDS = 4
MAX_EVALS = 10

def objective(params, n_folds = N_FOLDS):
    """Objective function for Logistic Regression Hyperparameter Tuning"""

    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evaluate based on ROC AUC
    mlflow.sklearn.autolog()
    with mlflow.start_run(nested=True):
        clf = LogisticRegression(**params,random_state=0,verbose =0)
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')

        # Extract the best score
        best_score = max(scores)

        # Loss must be minimized
        loss = 1 - best_score

        # Log the metric
        mlflow.log_metric(key="f1_experiment_score", value=best_score)

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

# Define parameter space
space = {
    'warm_start' : hp.choice('warm_start', [True, False]),
    'fit_intercept' : hp.choice('fit_intercept', [True, False]),
    'tol' : hp.uniform('tol', 0.00001, 0.0001),
    'C' : hp.uniform('C', 0.05, 3),
    'solver' : hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
    'max_iter' : hp.choice('max_iter', range(5,1000))
}

# Create experiment
mlflow.set_experiment("HyperOpt_Logsitic")

# Define Optimization Trials
tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()

with mlflow.start_run():
    best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 583, "status": "ok", "timestamp": 1630412600501, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="sGFqgn2oKZY3" outputId="3d7ec6ac-0171-42ea-99d3-ad6cf020a1f5"
best
```

<!-- #region id="cIp2cIsRKj1f" -->
<!-- #endregion -->

<!-- #region id="XtaCaRYtAfH2" -->
## Model registry
<!-- #endregion -->

<!-- #region id="TMyPBm8pLDdp" -->
<!-- #endregion -->
