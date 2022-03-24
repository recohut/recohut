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

# Auto Surprise Library

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 39223, "status": "ok", "timestamp": 1629793359475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="U2ex9ekEPqIr" outputId="8eb87e5b-1e73-41a6-d69f-6399aa78303b"
!pip install surprise
!pip install auto-surprise
```

<!-- #region id="4EkxNOjiSNyC" -->
## Quick start
<!-- #endregion -->

```python executionInfo={"elapsed": 703, "status": "ok", "timestamp": 1629793369375, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="eS57iiaSQL2r"
from surprise import Dataset
from auto_surprise.engine import Engine
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 3609863, "status": "ok", "timestamp": 1629796979217, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="E0tc-e_3QG1-" outputId="b487463c-8a7c-44eb-e93a-377195fc566f"
# Load the dataset
data = Dataset.load_builtin('ml-100k')

# Intitialize auto surprise engine
engine = Engine(verbose=True)

# Start the trainer
best_algo, best_params, best_score, tasks = engine.train(
    data=data, 
    target_metric='test_rmse', 
    cpu_time_limit=60 * 60, 
    max_evals=100
)
```

<!-- #region id="WqxkPNhFQTXy" -->
## Exploration
<!-- #endregion -->

```python id="_9mjNDoDQ99B"
import random
import numpy as np
import hyperopt
from surprise import Dataset
from auto_surprise.engine import Engine
```

```python id="f7KHVu2tRRue"
random.seed(123)
np.random.seed(123)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27967, "status": "ok", "timestamp": 1629793620569, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="i0APyBrzRDf1" outputId="0e2e3348-8aec-4e18-b879-d7ee405f65a3"
# Load the dataset
data = Dataset.load_builtin('ml-100k')
```

```python id="M0hl5N8YRECV"
# Intitialize auto surprise engine with random state set
engine = Engine(verbose=True, random_state=np.random.RandomState(123),
                algorithms=['svd', 'knn_basic', 'knn_baseline'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 902812, "status": "ok", "timestamp": 1629801039990, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="t019Tam7RmcU" outputId="53451165-75ac-4e30-d6d4-0a298b329d5b"
# start the optimization process
best_algo, best_params, best_score, tasks = engine.train(
    data=data,
    target_metric='test_rmse',
    cpu_time_limit=60*60*2,
    max_evals=100,
    hpo_algo=hyperopt.atpe.suggest
)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 571, "status": "ok", "timestamp": 1629801252688, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Hp3yk8MjSwKf" outputId="cc2150b3-35a3-420a-f17c-fb513676bd60"
print("Best algorithm: ", best_algo)
print("Best params: ", best_params)
print("Best score: ", best_score)
print("All tasks: ", tasks)
```

```python id="b9YUwoWKR28K"
# building the best model
best_model = engine.build_model(best_algo, best_params)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 619, "status": "ok", "timestamp": 1629801342804, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="xgWbvszMucN9" outputId="584b17c0-5028-462f-87b2-d55705e64481"
best_model
```
