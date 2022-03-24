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

```python id="axG_1Qy3-l7g" executionInfo={"status": "ok", "timestamp": 1628949907902, "user_tz": -330, "elapsed": 616, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-ffr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="j54l6OF3-uTO" executionInfo={"status": "ok", "timestamp": 1628949953569, "user_tz": -330, "elapsed": 34318, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8ac29118-a4e9-437c-9317-837526d4a56c"
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

```python colab={"base_uri": "https://localhost:8080/"} id="urJ5hnyU-uTS" executionInfo={"status": "ok", "timestamp": 1628950048095, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="60fe0417-a327-4b82-ed43-440394d990de"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="M0rNx77D-uTT" executionInfo={"status": "ok", "timestamp": 1628950081823, "user_tz": -330, "elapsed": 1634, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c06a7d7b-687e-4658-ccb0-8807869d2bf5"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="pCDj92Xg-uTT" executionInfo={"status": "ok", "timestamp": 1628949953575, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1af68b7f-f16c-4da8-95bc-329e895c0ea2"
import glob
files = sorted(glob.glob(f'/content/drive/MyDrive/Colab Notebooks/{project_name}*.ipynb'))
files
```

```python id="Be-JwaUz-uTU" executionInfo={"status": "ok", "timestamp": 1628949964578, "user_tz": -330, "elapsed": 1803, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from shutil import copy
for fpath in files[1:]:
    copy(fpath, './notebooks')
```

```sh colab={"base_uri": "https://localhost:8080/"} id="MWPPuwD8-uTU" executionInfo={"status": "ok", "timestamp": 1628949971695, "user_tz": -330, "elapsed": 5422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="527279e8-87fa-480a-b28d-89bd872014ea"
mkdir -p ./code/nbs
for file in ./notebooks/*.ipynb
do
    jupyter nbconvert --output-dir="./code/nbs" --to python "$file"
done
```

```python colab={"base_uri": "https://localhost:8080/"} id="TO9WrtZV-uTV" executionInfo={"status": "ok", "timestamp": 1628949982589, "user_tz": -330, "elapsed": 6603, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8a1c9851-8c64-4fb7-d34a-5aa686334028"
!sudo apt-get install -qq tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="CO_t8xsT-uTV" executionInfo={"status": "ok", "timestamp": 1628949982591, "user_tz": -330, "elapsed": 34, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dea1710b-0665-4e07-a2c4-66b524e750f9"
!tree -L 3 .
```

````python colab={"base_uri": "https://localhost:8080/"} id="oXSudHMS-uTW" executionInfo={"status": "ok", "timestamp": 1628950042169, "user_tz": -330, "elapsed": 432, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9c0ae75b-2ca1-44a4-c6be-dff9ed219f7d"
%%writefile README.md
# Facebook Friend Recommender

This is a friend recommendation system used on social media platforms (e.g. Facebook, Instagram, Twitter) to suggest friends/new connections based on common interests, workplace, common friends etc. using Graph Mining techniques. Here, we are given a social graph, i.e. a graph structure where nodes are individuals on social media platforms and a directed edges (or 'links') indicates that one person 'follows' the other, or are 'friends' on social media. Now, the task is to predict newer edges to be offered as 'friend suggestions'.

First we will load our dataset from Kaggle and perform exploratory data analysis on our given data set such as number of followers and followees of each person. Then we will generate some datapoints which were not present in our given data-set, since we have only class label 1 data. Then we will do some feature engineering on dataset like finding shortest path, kartz centrality, jaccard distances, page rank, preferential attachements etc. After performing exploratory data analysis and feature engineering, we will split whole dataset into train and test and perform random forest and xgboost taking f1-score as our metric. At the end we will plot confusion matrix and pretty-table for both algorithm and finf best hyperparameters.

**Problem statement** - Given a directed social graph, have to predict missing links to recommend users (Link Prediction in graph)

## Project structure
```
.
├── code
│   └── nbs
│       ├── reco-tut-ffr-01-ingestion.py
│       ├── reco-tut-ffr-02-ETL.py
│       ├── reco-tut-ffr-03-preprocessing.py
│       ├── reco-tut-ffr-04-feature-engineering.py
│       └── reco-tut-ffr-05-modeling.py
├── data
│   ├── bronze
│   │   ├── test.parquet.gzip
│   │   └── train.parquet.gzip
│   ├── gold
│   │   ├── katz.p
│   │   ├── page_rank.p
│   │   ├── storage_sample_stage1.h5
│   │   ├── storage_sample_stage2.h5
│   │   ├── storage_sample_stage3.h5
│   │   └── storage_sample_stage4.h5
│   └── silver
│       ├── X_test_neg.parquet.gzip
│       ├── X_test_pos.parquet.gzip
│       ├── X_train_neg.parquet.gzip
│       ├── X_train_pos.parquet.gzip
│       ├── y_test.parquet.gzip
│       └── y_train.parquet.gzip
├── graph_sample.pdf
├── LICENSE
├── model
├── notebooks
│   ├── reco-tut-ffr-01-ingestion.ipynb
│   ├── reco-tut-ffr-02-ETL.ipynb
│   ├── reco-tut-ffr-03-preprocessing.ipynb
│   ├── reco-tut-ffr-04-feature-engineering.ipynb
│   └── reco-tut-ffr-05-modeling.ipynb
└── README.md  
```
````

```python colab={"base_uri": "https://localhost:8080/"} id="fSarlrrE-uTX" executionInfo={"status": "ok", "timestamp": 1628950101328, "user_tz": -330, "elapsed": 8794, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7683bd85-dc34-4c5a-9d06-25dc3fa85fd8"
xproject_name = "reco-nb-stage"; xbranch = "queued"; xaccount = "recohut"
xproject_path = os.path.join('/content', xproject_name)

if not os.path.exists(xproject_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + xproject_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{xaccount}"/"{xproject_name}".git
    !git pull origin "{xbranch}"
else:
    %cd "{xproject_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="Br2neLgR-uTY" executionInfo={"status": "ok", "timestamp": 1628950103745, "user_tz": -330, "elapsed": 2442, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="15fc738c-8067-4486-b3ec-4afbe9b0b5e1"
!git checkout -b queued
!cp $project_path/notebooks/*.ipynb ./_notebooks
!git add . && git commit -m 'commit' && git push origin queued
```

```python id="GSmDh8z9_dkR"

```
