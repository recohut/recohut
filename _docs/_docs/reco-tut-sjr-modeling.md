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

```python colab={"base_uri": "https://localhost:8080/"} id="F5rLoSxIocB-" executionInfo={"status": "ok", "timestamp": 1628192964424, "user_tz": -330, "elapsed": 2610, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ad8865e3-1e4b-451c-d284-d83ef5f4a511"
import os
project_name = "reco-tut-sjr"; branch = "main"; account = "sparsh-ai"
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

```python id="yxDicnp22_YA" executionInfo={"status": "ok", "timestamp": 1628194738198, "user_tz": -330, "elapsed": 673, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import spacy
```

```python colab={"base_uri": "https://localhost:8080/"} id="GkOUAxU07u07" executionInfo={"status": "ok", "timestamp": 1628194595843, "user_tz": -330, "elapsed": 682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="27c06b7a-af74-41d9-9ca8-0deb82c1adbf"
# !python -m spacy download en_core_web_lg
!ls /usr/local/lib/python3.7/dist-packages/en_core_web_lg
```

```python id="Uhc8yCDn753U" executionInfo={"status": "ok", "timestamp": 1628194621341, "user_tz": -330, "elapsed": 12712, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
nlp = spacy.load('/usr/local/lib/python3.7/dist-packages/en_core_web_lg/en_core_web_lg-2.2.5')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="HarDoB_U3cvq" executionInfo={"status": "ok", "timestamp": 1628193890759, "user_tz": -330, "elapsed": 474, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cc66f714-2bbc-4a7b-b4ad-c2f4050a7078"
df_jobs = pd.read_pickle('./data/silver/jobs.p', compression='gzip')
df_jobs = df_jobs.reset_index(drop=True)
df_jobs.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ug32fZ5M33xx" executionInfo={"status": "ok", "timestamp": 1628193909428, "user_tz": -330, "elapsed": 624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="43ff38aa-52d8-41b8-89fd-7ce415f430f7"
df_users = pd.read_pickle('./data/silver/applicants.p', compression='gzip')
df_users = df_users.reset_index(drop=True)
df_users.head()
```

<!-- #region id="Y0t7AY1c4hhZ" -->
## Selecting test user
<!-- #endregion -->

```python id="7eH1ykkV2LMH" executionInfo={"status": "ok", "timestamp": 1628193915856, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_recommendation(top, df_all, scores):
  recommendation = pd.DataFrame(columns = ['ApplicantID', 'JobID',  'title', 'score'])
  count = 0
  for i in top:
      recommendation.at[count, 'ApplicantID'] = u
      recommendation.at[count, 'JobID'] = df_all['Job.ID'][i]
      recommendation.at[count, 'title'] = df_all['Title'][i]
      recommendation.at[count, 'score'] =  scores[count]
      count += 1
  return recommendation
```

```python colab={"base_uri": "https://localhost:8080/", "height": 80} id="97VSGKYNi-PO" executionInfo={"status": "ok", "timestamp": 1628193916575, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="735bf60c-5884-4c38-c0e3-d54cf2ed3768"
u = 10001
index = np.where(df_users['Applicant_id'] == u)[0][0]
user_q = df_users.iloc[[index]]
user_q
```

<!-- #region id="RLBeZTFGi9zK" -->
## Model 1 - TFIDF
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QwziWFUki9zP" executionInfo={"status": "ok", "timestamp": 1628193913535, "user_tz": -330, "elapsed": 443, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="79b15f7b-12e6-423d-bf91-5602fa846fb5"
#initializing tfidf vectorizer
##This is a technique to quantify a word in documents, 
#we generally compute a weight to each word which signifies the importance of the word in the document and corpus. 
##This method is a widely used technique in Information Retrieval and Text Mining.
tfidf_vectorizer = TfidfVectorizer()

tfidf_jobid = tfidf_vectorizer.fit_transform((df_jobs['text'])) #fitting and transforming the vector
tfidf_jobid
```

<!-- #region id="gTuiU3-rldbv" -->
Computing cosine similarity using tfidf
<!-- #endregion -->

```python id="nFKWoQlTi-RZ" colab={"base_uri": "https://localhost:8080/", "height": 359} executionInfo={"status": "ok", "timestamp": 1628193936006, "user_tz": -330, "elapsed": 5909, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ca4d7dd3-9aee-45e0-8c9d-a62cf3d514af"
user_tfidf = tfidf_vectorizer.transform(user_q['text'])
cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x), tfidf_jobid)
output2 = list(cos_similarity_tfidf)

top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
list_scores = [output2[i][0][0] for i in top]
get_recommendation(top, df_jobs, list_scores)
```

<!-- #region id="-0Fqp_y2i-Rg" -->
## Model 2 - CountVectorizer
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="skW2QC8ni-Rm" executionInfo={"status": "ok", "timestamp": 1628194024517, "user_tz": -330, "elapsed": 785, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="96bc1f60-10ed-461a-cf19-7acc62403520"
count_vectorizer = CountVectorizer()
count_jobid = count_vectorizer.fit_transform((df_jobs['text'])) #fitting and transforming the vector
count_jobid
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="9BiidHnc7O8A" executionInfo={"status": "ok", "timestamp": 1628194122999, "user_tz": -330, "elapsed": 7548, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6cdcfcf-5f88-4c33-8fad-b4020946338b"
user_count = count_vectorizer.transform(user_q['text'])
cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x),count_jobid)
output2 = list(cos_similarity_countv)

top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
list_scores = [output2[i][0][0] for i in top]
get_recommendation(top, df_jobs, list_scores)
```

<!-- #region id="H5j-pPnF8W9p" -->
## Model 3 - Spacy
<!-- #endregion -->

<!-- #region id="6DR9KWzZ2bkM" -->
Transform the copurs text to the *spacy's documents* 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 51} id="68Jj02EY6McH" outputId="0c13a658-b2f6-41d9-9e89-f8cf238ab91f"
%%time
list_docs = []
for i in range(len(df_jobs)):
  doc = nlp("u'" + df_jobs['text'][i] + "'")
  list_docs.append((doc,i))
print(len(list_docs))
```

```python id="YcgHBSzH3-UP"
def calculateSimWithSpaCy(nlp, df, user_text, n=6):
    # Calculate similarity using spaCy
    list_sim =[]
    doc1 = nlp("u'" + user_text + "'")
    for i in df.index:
      try:
            doc2 = list_docs[i][0]
            score = doc1.similarity(doc2)
            list_sim.append((doc1, doc2, list_docs[i][1],score))
      except:
        continue

    return  list_sim   
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="EGmD1vaI5T_h" outputId="fb803467-ce16-4593-b9ae-ae9cbddc14ca"
user_q.text[186]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 51} id="8izD0IrrGX_g" outputId="de3769ac-7e52-4979-878e-756b46a90730"
df3 = calculateSimWithSpaCy(nlp, df_jobs, user_q.text[186], n=15)
df_recom_spacy = pd.DataFrame(df3).sort_values([3], ascending=False).head(10)
df_recom_spacy.reset_index(inplace=True)

index_spacy = df_recom_spacy[2]
list_scores = df_recom_spacy[3]
```

<!-- #region id="3tGb5yvF3iFd" -->
Top recommendations using Spacy
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="vXewK2XASV9J" outputId="1fdd1540-843b-4afb-c2b8-5f5cd0acee66"
get_recommendation(index_spacy, df_jobs, list_scores)
```

<!-- #region id="7b2uGpnn8XdU" -->
## Model 4 - KNN
<!-- #endregion -->

```python id="TRXObP8P8sgC" executionInfo={"status": "ok", "timestamp": 1628194743864, "user_tz": -330, "elapsed": 619, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
n_neighbors = 11
KNN = NearestNeighbors(n_neighbors, p=2)
KNN.fit(tfidf_jobid)
NNs = KNN.kneighbors(user_tfidf, return_distance=True) 
```

```python colab={"base_uri": "https://localhost:8080/"} id="HIMqpa4KG1Yc" executionInfo={"status": "ok", "timestamp": 1628194745338, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c1c023bd-5392-4f76-eb57-3aa26eda532e"
NNs[0][0][1:]
```

<!-- #region id="sEEa3CNX3J1D" -->
The top recommendations using KNN
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="OUmdrNOlHD-2" executionInfo={"status": "ok", "timestamp": 1628194753284, "user_tz": -330, "elapsed": 699, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9b97792d-3c02-4766-b367-4367e50442a7"
top = NNs[1][0][1:]
index_score = NNs[0][0][1:]

get_recommendation(top, df_jobs, index_score)
```
