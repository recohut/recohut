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

<!-- #region id="FIOyUcOgRmGf" -->
# Ravelry Pattern Recommender
> Recommend Wool knitted desings (artifacts) to users

- toc: true
- badges: true
- comments: true
- categories: [SVD, Surprise, API, Art&Culture]
- image:
<!-- #endregion -->

<!-- #region id="nwRjVKbeKNVT" -->
<!-- #endregion -->

<!-- #region id="7F7Ixp_qLYEG" -->
[Ravelry](https://www.ravelry.com/about) describes itself as a place for knitters, crocheters, designers, spinners, weavers and dyers to keep track of their yarn, tools, project and pattern information, and look to others for ideas and inspiration.
<!-- #endregion -->

<!-- #region id="vmY1dm8CORIP" -->
## Setup
<!-- #endregion -->

```python id="1r1i9yf6Oa7K"
!pip install -q surprise
```

```python id="Sz5ifkA8OYvb"
import pandas as pd
import numpy as np
import datetime as dt

import surprise
from surprise.prediction_algorithms import *
from surprise import Reader, Dataset
from surprise import SVD, accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')
```

```python colab={"base_uri": "https://localhost:8080/"} id="gXowNBAnmDh9" outputId="b18f0b69-7e47-4302-cb10-c3c86ad26a99"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv
```

```python id="d9kuIz7HmDh_"
# Poetry is a tool for dependency management and packaging in Python. 
# It allows you to declare the libraries your project depends on and it will manage (install/update) them for you.
# https://python-poetry.org/docs/basic-usage/
# !curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
# !/root/.local/bin/poetry --version
# !/root/.local/bin/poetry new poetry-demo
# %cd poetry-demo
# !/root/.local/bin/poetry install
# !/root/.local/bin/poetry add numpy
```

<!-- #region id="XvWdTkkYQ0NY" -->
## What are patterns?
<!-- #endregion -->

<!-- #region id="siXUALtfRIJt" -->
<!-- #endregion -->

<!-- #region id="7Ev4xaxlRbnV" -->
> youtube: https://youtu.be/ybEClAPFF8M
<!-- #endregion -->

<!-- #region id="tF7mJenvOU04" -->
## Data
<!-- #endregion -->

<!-- #region id="UriwzI9lNJVT" -->
### Data Fetching from API (optional)
<!-- #endregion -->

```python id="_mk9kmfFKbKO"
# import pandas as pd
# import requests
# import json
# import random
# import numpy

# with open('creds.json') as f:
#     creds = json.load(f)

# users = []

# for i in random.sample(range(1, 12000000), 50000):
#     try:
#         url ='https://api.ravelry.com/people/' + str(i) +'.json'
#         response = requests.get(url, auth=(creds['id'], creds['key']))
#         users.append(response.json()['user']['username'])
#     except ValueError: 
#         pass
#     if len(set(users)) >10000:
#         break

# parsed_data = []

# for i, user in enumerate(users[9935:]): 
    
#     url ='https://api.ravelry.com/projects/' + user + '/list.json?sort=completed_'
#     response = requests.get(url, auth=(creds['id'], creds['key']))
    
#     try:
#         for project in response.json()['projects']:
#             if project['craft_name'] == 'Knitting': 
#                 if project['pattern_id'] != None:
#                     pattern_url ='https://api.ravelry.com/patterns.json?ids=' + str(int(project['pattern_id']))
#                     pattern_response = requests.get(pattern_url, auth=(creds['id'], creds['key']))
#                     project_tuple = (user, project['completed'], project['rating'], project['status_name'], 
#                                      project['pattern_id'],
#                                      pattern_response.json()['patterns'][str(int(project['pattern_id']))]['rating_average'],
#                                      pattern_response.json()['patterns'][str(int(project['pattern_id']))]['rating_count'])
#                     parsed_data.append(project_tuple)   
            
#     except ValueError:
#         pass

#     print(i, len(parsed_data))

# df = pd.DataFrame(parsed_data, columns= ['user', 'completed', 'rating', 'status', 'pattern_id', 'average_rating', 'rating_count'])

# finished_projects = df[df['status'] == 'Finished']
# finished_projects.to_csv('ravelry_interactions.csv', index=False)
```

<!-- #region id="eW8b8InXSPAq" -->
### Direct loading from drive
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="sTgvoAn-OPY2" outputId="44f81d91-4abb-4849-bc96-48b883d8e468"
df = pd.read_csv('https://raw.githubusercontent.com/recohut/reco-data/ravelry/ravelry/v1/ravelry_interactions.csv')
df
```

<!-- #region id="MjAup1CpSTxb" -->
## Preprocessing
<!-- #endregion -->

```python id="N9l47cqiO3oN" colab={"base_uri": "https://localhost:8080/", "height": 419} outputId="77416481-dc14-4fc3-b26f-0439aa2a980d"
df_drop_nans = df[['user', 'pattern_id', 'rating']].dropna(subset = ['rating'])
df_drop_nans
```

```python id="vLxYSS1GO3oO" colab={"base_uri": "https://localhost:8080/", "height": 142} outputId="049b21de-3d78-40f5-8219-889dd19f10e3"
df_drop_nans.describe(include='all').T
```

```python id="hNNOqLgGO3oP" colab={"base_uri": "https://localhost:8080/", "height": 419} outputId="a1ce6f18-68e4-496f-b291-f28a20346a65"
df_replace_nans = df[['user', 'pattern_id', 'rating', 'average_rating']]
rating_replace_nans = df_replace_nans['rating'].fillna(df_replace_nans['average_rating'])
df_replace_nans['rating'] = rating_replace_nans
df_replace_nans.drop(columns = 'average_rating', inplace = True)
df_replace_nans
```

<!-- #region id="g7SkotwASXDO" -->
## Surprise Dataset
<!-- #endregion -->

```python id="CQDgqT1QO3oQ"
reader = Reader()
data_replace = Dataset.load_from_df(df_replace_nans, reader)
data_drop = Dataset.load_from_df(df_drop_nans, reader)
```

```python id="h5PaCaUQO3oQ"
drop_trainset, drop_testset = train_test_split(data_drop, test_size=0.25)
replace_trainset, replace_testset = train_test_split(data_replace, test_size=0.25)
```

```python id="viTrkkNxO3oQ" colab={"base_uri": "https://localhost:8080/"} outputId="6a22f2ba-8a25-4234-c1c6-3884b8698528"
drop_trainset.global_mean
```

<!-- #region id="Fp7kMisbSZqy" -->
## SVD model
<!-- #endregion -->

```python id="r8sN2QQmO3oR" colab={"base_uri": "https://localhost:8080/"} outputId="7bc9f777-8c4d-4450-bc94-3a6387a9b3f0"
algo = SVD(n_factors = 50, n_epochs = 45, lr_all = 0.004, reg_all = 0.2)
algo.fit(drop_trainset)
```

```python id="4aYUffujO3oS"
predictions = algo.test(drop_testset)
```

```python id="HkkxVTKZO3oS" colab={"base_uri": "https://localhost:8080/"} outputId="a65302c8-6a56-491a-9340-6e6967fc243f"
accuracy.rmse(predictions)
```

<!-- #region id="I1ual6JfSdVa" -->
## Hyperparameter search
<!-- #endregion -->

```python id="eRfVa1_8O3oT" colab={"base_uri": "https://localhost:8080/"} outputId="aa554557-f3fe-4ec9-da00-b6d7a126d546"
param_grid = {'n_factors':[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
              'n_epochs': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], 
              'lr_all': [0.002, 0.003, 0.004, 0.005],
              'reg_all': [0.2, 0.3, 0.4, 0.5, 0.6]}

gs_model = GridSearchCV(SVD,
                        param_grid=param_grid,
                        n_jobs = -1,
                        joblib_verbose=5)

gs_model.fit(data_drop)

gs_model.best_params
```

<!-- #region id="6K4HAML7Sg3G" -->
## Inference (rating prediction)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="l3Vzqa8EjvRC" outputId="7d088e26-8c2e-4b78-8575-f9e151e0d48f"
algo = SVD(**gs_model.best_params['rmse'])
algo.fit(drop_trainset)
predictions = algo.test(drop_testset)
accuracy.rmse(predictions)
```

```python id="Zhj3FhxgO3oV"
predictions_df = pd.DataFrame({"user": [prediction.uid for prediction in predictions],
                   "item": [prediction.iid for prediction in predictions],
                   "actual": [prediction.r_ui for prediction in predictions],
                   "estimated" :[prediction.est for prediction in predictions]})
```

```python id="sdPRZ6qGO3oV" colab={"base_uri": "https://localhost:8080/", "height": 297} outputId="dd30950a-58dd-45fd-a56d-cebdc9a911d1"
predictions_df[predictions_df['user'] == 'Ona'].describe()
```

```python id="i8IWyRjAO3oW" colab={"base_uri": "https://localhost:8080/", "height": 855} outputId="9f9257ea-4676-4f3b-8102-36f070ddad3a"
df[df['user'] == "Ona"]
```

```python id="W50r3F1cO3oW" colab={"base_uri": "https://localhost:8080/"} outputId="b056c023-e59e-44b5-caf9-15050d4e2f29"
algo.predict('Ona', 1)
```

<!-- #region id="3UiBUobDSzt9" -->
## References
1. https://github.com/clareadunne/PatternRecommender `code`
2. https://www.ravelry.com/account/login `site`
3. [Googoe Images search service](https://www.google.com/search?q=ravelry&rlz=1C1GCEA_enIN909IN909&sxsrf=ALeKk01ydUvKpgGnU5d_WLs-CPtDmfh2wQ:1626942092877&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjg7I_Mn_bxAhWpzDgGHe6_BjgQ_AUoAXoECAEQAw&biw=1366&bih=657) `site`
4. https://youtu.be/ybEClAPFF8M `video`
<!-- #endregion -->
