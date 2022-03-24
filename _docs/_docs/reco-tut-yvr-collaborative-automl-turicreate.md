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

```python colab={"base_uri": "https://localhost:8080/"} id="RipmMuMXcZzB" executionInfo={"status": "ok", "timestamp": 1628102076009, "user_tz": -330, "elapsed": 2655, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ee858308-b827-4692-91c3-222a2213118e"
import os
project_name = "reco-tut-yvr"; branch = "main"; account = "sparsh-ai"
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

```python id="m6-aexJOcnmB"
!git add . && git commit -m 'commit' && git push origin main
```

```python id="N5BTBwKvcZzD"
!pip install turicreate
```

```python id="aU8qHs_Kc0Ds" executionInfo={"status": "ok", "timestamp": 1628102575544, "user_tz": -330, "elapsed": 419, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd 
import nltk
from nltk.corpus import wordnet as wn
import turicreate as tc
from sklearn.utils import shuffle
```

```python id="UZl7C4tjeWle"
nltk.download('wordnet')
```

```python colab={"base_uri": "https://localhost:8080/"} id="sljEcEf9dvov" executionInfo={"status": "ok", "timestamp": 1628102452278, "user_tz": -330, "elapsed": 2424, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ce81890b-fcdd-41d4-9acc-744ec1641ad6"
listnames = []

for i,j in enumerate(wn.synsets('music')):
    listnames.append(j.lemma_names())
print (listnames)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="GzHi3jlGeEOW" executionInfo={"status": "ok", "timestamp": 1628102526696, "user_tz": -330, "elapsed": 970, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="743bafa1-a397-49fd-e733-8841166f39a5"
df = pd.read_csv('./data/bronze/data.csv', encoding='utf-8', index_col=0).reset_index(drop=True)
df = df.drop_duplicates(['v_title'])
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Kj0MXpljeINc" executionInfo={"status": "ok", "timestamp": 1628102536050, "user_tz": -330, "elapsed": 495, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="93baa99f-5e06-4f2f-d80f-882e13a6b633"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="w44i_ocVeSJn" executionInfo={"status": "ok", "timestamp": 1628102609503, "user_tz": -330, "elapsed": 1222, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="77cea4ed-16eb-4bd7-87ea-bd389310a921"
newdf = pd.read_csv('./data/bronze/third.csv', encoding='utf-8')
newdf = shuffle(newdf) #Shuffling the dataset in order to randomize the data
newdf
```

```python id="tOrRIVXDekEg" executionInfo={"status": "ok", "timestamp": 1628102663816, "user_tz": -330, "elapsed": 608, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train = newdf.loc[:250,:]
test = newdf.loc[250:,:]
```

```python id="hRDRB35kexho" executionInfo={"status": "ok", "timestamp": 1628102666819, "user_tz": -330, "elapsed": 669, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train_data = tc.SFrame(train)
test_data = tc.SFrame(test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 85} id="n_0liepseyRo" executionInfo={"status": "ok", "timestamp": 1628102694638, "user_tz": -330, "elapsed": 728, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="87fdd3af-0531-4d6f-a8e9-c4815f3b0608"
popularity_model = tc.popularity_recommender.create(train_data, user_id='users', item_id='v_title', target='Liked')
```

```python colab={"base_uri": "https://localhost:8080/"} id="RVGlBOmNe7Ib" executionInfo={"status": "ok", "timestamp": 1628102771161, "user_tz": -330, "elapsed": 730, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ba1586e6-9966-4324-cd51-05aabbfca7d1"
popularity_recomm = popularity_model.recommend(users=newdf.users.unique(), k=10)
popularity_recomm.print_rows(num_rows=25)
```

```python colab={"base_uri": "https://localhost:8080/"} id="FtkgqMGkfLwN" executionInfo={"status": "ok", "timestamp": 1628102826541, "user_tz": -330, "elapsed": 811, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="365280b5-57e5-4b78-9316-c75ebef89c3c"
train.groupby(by='v_title')['Liked'].sum().sort_values(ascending=False).head(20)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} id="KQRmyJ4ifTjw" executionInfo={"status": "ok", "timestamp": 1628102904827, "user_tz": -330, "elapsed": 2148, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e47c934c-686d-4056-cf96-a94a146f12e7"
item_sim_model = tc.item_similarity_recommender.create(train_data, user_id='users', item_id='v_title', target='Liked', similarity_type='cosine')
```

```python colab={"base_uri": "https://localhost:8080/"} id="wFbu7OlufsBc" executionInfo={"status": "ok", "timestamp": 1628102935856, "user_tz": -330, "elapsed": 977, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3bcaad21-9dea-4dd5-ba74-c6fcce2d0832"
item_sim_recomm = item_sim_model.recommend(users=newdf.users.unique(), k=5)
item_sim_recomm.print_rows(num_rows=25)
```

```python colab={"base_uri": "https://localhost:8080/"} id="eKnZyxDxfzlY" executionInfo={"status": "ok", "timestamp": 1628102970913, "user_tz": -330, "elapsed": 3563, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="487efe0f-b666-4049-a4f6-c32b90909ab8"
model_performance = tc.recommender.util.compare_models(test_data, [popularity_model, item_sim_model])
```
