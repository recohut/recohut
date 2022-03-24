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

```python id="86MgMsi_GD70" executionInfo={"status": "ok", "timestamp": 1628944874170, "user_tz": -330, "elapsed": 534, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-asr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

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

```python id="Y8Zw_6uTGD71" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628254775718, "user_tz": -330, "elapsed": 1750, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ae0da84c-ba4c-485e-80ed-329b07680b67"
!cd /content && git clone https://github.com/caiomiyashiro/RecommenderSystemsNotebooks.git
```

```python id="2MN38rKgjhWg"
!mv ./papers ./docs
```

```python colab={"base_uri": "https://localhost:8080/"} id="pXWJ6RWXjvEx" executionInfo={"status": "ok", "timestamp": 1628944614332, "user_tz": -330, "elapsed": 1620, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b2612233-1532-40ab-d4c2-080daef67514"
!git add . && git commit -m 'commit' && git push origin main
```

```python colab={"base_uri": "https://localhost:8080/"} id="jZMPPKgMojQ_" executionInfo={"status": "ok", "timestamp": 1628944119244, "user_tz": -330, "elapsed": 2000, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0a877db6-291b-489d-8e56-ec6789805805"
import glob
files = sorted(glob.glob(f'/content/drive/MyDrive/Colab Notebooks/{project_name}*.ipynb'))
files
```

```python id="vyxlbfMsojRA" executionInfo={"status": "ok", "timestamp": 1628944144794, "user_tz": -330, "elapsed": 1536, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from shutil import copy
for fpath in files[1:]:
    copy(fpath, './notebooks')
```

```sh id="91d4gVMEo53Z"
for file in ./notebooks/*.ipynb
do
    jupyter nbconvert --output-dir="./code/nbs" --to python "$file"
done
```

```python id="88ZHhcxgojRB"
!sudo apt-get install -qq tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="7T4vIE-jp1__" executionInfo={"status": "ok", "timestamp": 1628944491728, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ac34fd9d-492f-41c3-dd6b-be0b7bfb7e57"
!tree -L 3 .
```

````python colab={"base_uri": "https://localhost:8080/"} id="tXkaZK-PojRC" executionInfo={"status": "ok", "timestamp": 1628944595423, "user_tz": -330, "elapsed": 495, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="545df82c-81b9-452d-ee09-3bd05b5f1223"
%%writefile README.md
# Amazon Stationary Recommender

## Project structure
```
.
├── code
│   ├── __init__.py
│   ├── metrics.py
│   └── nbs
│       ├── reco-tut-asr-01-evaluation-version-1.py
│       ├── reco-tut-asr-01-evaluation-version-2.py
│       ├── reco-tut-asr-99-01-non-personalised-and-stereotyped-recommendation.py
│       ├── reco-tut-asr-99-02-recommendation-metrics-rating-evaluation.py
│       ├── reco-tut-asr-99-03-content-based-recommendations.py
│       ├── reco-tut-asr-99-04-user-user-cf.py
│       ├── reco-tut-asr-99-05-item-item-cf.py
│       ├── reco-tut-asr-99-06-similarity-metrics-for-cf.py
│       ├── reco-tut-asr-99-07-offline-evaluation.py
│       ├── reco-tut-asr-99-08-matrix-factorization.py
│       ├── reco-tut-asr-99-09-hybrid-recommendations.py
│       └── reco-tut-asr-99-10-metrics-calculation.py
├── data
│   ├── bronze
│   │   ├── items.csv
│   │   ├── other
│   │   └── ratings.csv
│   ├── gold
│   │   ├── cbf.csv
│   │   ├── item-item.csv
│   │   ├── mf.csv
│   │   ├── pers-bias.csv
│   │   └── user-user.csv
│   └── silver
│       ├── items.csv
│       └── ratings.csv
├── docs
│   ├── notes
│   │   ├── Month 1 - HW1 Notes.pdf
│   │   ├── Month 1 - HW 2 Notes.pdf
│   │   ├── Month 2 - HW 1 Notes.pdf
│   │   ├── Month 2 - HW2 Notes.pdf
│   │   ├── Month 3 - HW1 Notes.pdf
│   │   ├── Month 3 - HW2 Quizz Long.pdf
│   │   └── Month 4 - HW1 Notes.pdf
│   ├── papers
│   │   ├── An Algorithmic Framework for Performing Collaborative Filtering.pdf
│   │   ├── A Survey of Serendipity in Recommender Systems.pdf
│   │   ├── Collaborative Recommendations Using Item-to-Item Similarity Matrix.pdf
│   │   ├── Evaluating Collaborative Filtering Recommender Systems.pdf
│   │   ├── Explaining Collaborative Filltering Recommendations.pdf
│   │   ├── Improving Recommendation Lists Through Topic Diversification.pdf
│   │   ├── Is Seeing Believing - How Recommender Systems Interfaces bias User's Opinions.pdf
│   │   └── Item-Based Collaborative Filtering Algorithm.pdf
│   └── slides
│       ├── Coursera - 0 - Slides Introduction to RS.pdf
│       ├── Coursera 10 - Capstone Assignment.pdf
│       ├── Coursera - 1 - Non-Personalised and Stereotyped Rec.pdf
│       ├── Coursera - 2 - Content Based Recommendation.pdf
│       ├── Coursera - 3 - Slides User User Collaborative Filtering.pdf
│       ├── Coursera - 4 - Slides Item Item Collaborative Filtering.pdf
│       ├── Coursera - 5 - Additional Item and List‐Based Metrics.pdf
│       ├── Coursera - 6 - Top-N Protocols and Unary Data.pdf
│       ├── Coursera - 7 - Online-Evaluation-and-User-Studies.pdf
│       ├── Coursera - 8 - AB-Testing.pdf
│       └── Coursera - 9 - Evaluation-Cases.pdf
├── images
│   ├── notebook1_image1.jpeg
│   ├── notebook1_image2.jpg
│   ├── notebook2_image1.jpg
│   ├── notebook2_image2.jpg
│   ├── notebook2_image3.png
│   ├── notebook3_image1.jpg
│   ├── notebook4_image1.png
│   ├── notebook5_image1.jpeg
│   ├── notebook5_image2.png
│   ├── notebook6_image1.png
│   ├── notebook6_image2.png
│   ├── notebook6_image3.png
│   ├── notebook7_image1.png
│   ├── notebook7_image20.png
│   ├── notebook7_image3.png
│   ├── notebook7_image4.png
│   ├── notebook7_image50.png
│   ├── notebook7_image6.png
│   └── notebook7_image7.png
├── LICENSE
├── model
└── notebooks
    ├── reco-tut-asr-01-evaluation-version-1.ipynb
    ├── reco-tut-asr-01-evaluation-version-2.ipynb
    ├── reco-tut-asr-99-01-non-personalised-and-stereotyped-recommendation.ipynb
    ├── reco-tut-asr-99-02-recommendation-metrics-rating-evaluation.ipynb
    ├── reco-tut-asr-99-03-content-based-recommendations.ipynb
    ├── reco-tut-asr-99-04-user-user-cf.ipynb
    ├── reco-tut-asr-99-05-item-item-cf.ipynb
    ├── reco-tut-asr-99-06-similarity-metrics-for-cf.ipynb
    ├── reco-tut-asr-99-07-offline-evaluation.ipynb
    ├── reco-tut-asr-99-08-matrix-factorization.ipynb
    ├── reco-tut-asr-99-09-hybrid-recommendations.ipynb
    └── reco-tut-asr-99-10-metrics-calculation.ipynb
```
````

```python colab={"base_uri": "https://localhost:8080/"} id="KFX8sMAqqfTy" executionInfo={"status": "ok", "timestamp": 1628945004336, "user_tz": -330, "elapsed": 953, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b4f2a4c4-6d97-40eb-81af-f7c2278a5a45"
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

```python id="3u596jPprTGv"
!git checkout -b queued
!cp $project_path/notebooks/*.ipynb ./_notebooks
!git add . && git commit -m 'commit' && git push origin queued
```

```python id="SwReXirOrZwz" executionInfo={"status": "ok", "timestamp": 1628945272636, "user_tz": -330, "elapsed": 726, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}

```
