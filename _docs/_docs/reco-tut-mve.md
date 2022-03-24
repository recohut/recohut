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

```python id="BGgF7mbDpqo8" executionInfo={"status": "ok", "timestamp": 1630019691592, "user_tz": -330, "elapsed": 1039, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mve"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="gLlY3Kntpldf" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630019693580, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fb15aa30-d2a1-47f6-9b4b-7aea7519c8a8"
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

```python id="Y0k-sht3pldj" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630020855183, "user_tz": -330, "elapsed": 793, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="606a9d15-e3a9-4218-9228-3ae4c199a3b5"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="vceeBfALqf3U" executionInfo={"status": "ok", "timestamp": 1630020861463, "user_tz": -330, "elapsed": 1603, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ea6d98e4-868f-4298-a373-209690658fb2"
!git pull --rebase origin "{branch}"
```

```python id="zbD-LYR9pldj" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630020871407, "user_tz": -330, "elapsed": 4824, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f6c3d124-0d90-4a98-ea0e-a72175b78403"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="eDqQmJHjp4-w" -->
---
<!-- #endregion -->

<!-- #region id="8GKtSg3_uBX_" -->
## Web Scraping Script
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} cellView="form" id="LIT9rfXfp43X" executionInfo={"status": "ok", "timestamp": 1630018529668, "user_tz": -330, "elapsed": 731, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="143773c0-467f-4e12-b78b-c10ec75f64cb"
#@title
%%writefile ./code/imdb_scraping.py
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup


headers = ({'User-Agent':
            'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36',
           'Accept-Language': 'en-US, en;q=0.5'})


movie_name = []
year = []
time=[]
rating=[]
metascore =[]
director=[]
votes = []
gross = []
description = []
genre=[]
cast=[]

pages = np.arange(1,1000,50)
#https://www.imdb.com/search/title/?title_type=feature&primary_language=en
#https://www.imdb.com/search/title/?title_type=feature&primary_language=en&start=51&ref_=adv_nxt
for page in pages:
   
    page = requests.get("https://www.imdb.com/search/title/?title_type=feature&primary_language=en&start="+str(page)+"&ref_=adv_nxt")
    soup = BeautifulSoup(page.text, 'html.parser')
    movie_data = soup.findAll('div', attrs = {'class': 'lister-item mode-advanced'})
    for store in movie_data:
        name = store.h3.a.text
        movie_name.append(name)
        
        year_of_release = store.h3.find('span', class_ = "lister-item-year text-muted unbold").text.replace('(', '')
        year_of_release=year_of_release.replace(')','')
        year.append(year_of_release)
        
        runtime = store.p.find("span", class_ = 'runtime').text if store.find('span', class_ = "runtime") else "NA"
        time.append(runtime)
        
        gen = store.p.find("span", class_ = 'genre').text
        genre.append(gen)
        
        rate = store.find('div', class_ = "inline-block ratings-imdb-rating").text.replace('\n', '') if store.find('div', class_ = "inline-block ratings-imdb-rating") else "NA"
        rating.append(rate)
        #rate = store.find('div', class_ = "ratings-bar").find('strong').text.replace('\n', '')
        #rating.append(rate)
        
        meta = store.find('span', class_ = "metascore").text if store.find('span', class_ = "metascore") else "NA"#if meta score not present then *
        
        metascore.append(meta)
        
        #dire=store.find('p',class_ = "metascore")
        dire=store.find('p',class_='').find_all('a')[0].text
        
        director.append(dire)
        cas=store.find('p',class_='').find_all('a')[1].text
        cas1=store.find('p',class_='').find_all('a')[2].text
        cas2=store.find('p',class_='').find_all('a')[3].text
        cas3=cas+','+cas1+','+cas2
        cast.append(cas3)
        
        
        value = store.find_all('span', attrs = {'name':'nv'}) if store.find_all('span', attrs = {'name':'nv'}) else 'NA'
        vote = value[0].text if store.find_all('span', attrs = {'name':'nv'}) else 'NA'

        #vote = value[0].text if len(value)>1 else 'NA'
        votes.append(vote)
        
        #grosses = value[1].text if len(value)>1 else 'NA'
        #gross.append(grosses)
        
      
        describe = store.find_all('p', class_ = 'text-muted')
        description_ = describe[1].text.replace('\n', '') if len(describe) >1 else 'NA'
        description.append(description_)
        
#dataframe
movie_list = pd.DataFrame({ "Movie Name": movie_name, "Year of Release" : year, "Watch Time": time,"Genre":genre,"Movie Rating": rating, "Metascore of movie": metascore,"Director":director,"Cast":cast,"Votes" : votes,"Description": description})
movie_list.to_excel("../data/bronze/imdb_scraped.xlsx")
```

<!-- #region id="aumUNXHjrh8R" -->
## Content-based Recommender Model
<!-- #endregion -->

```python id="aSziQo7Lrher" executionInfo={"status": "ok", "timestamp": 1630018622343, "user_tz": -330, "elapsed": 1649, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
```

```python colab={"base_uri": "https://localhost:8080/"} id="XHwB8pRTrpyx" executionInfo={"status": "ok", "timestamp": 1630019022738, "user_tz": -330, "elapsed": 1322, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4a54692b-c4db-493b-e8fc-4e19d7b485b9"
data = pd.read_excel("./data/bronze/imdb_scraped.xlsx")
data.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="1EiCQRcpr3Xp" executionInfo={"status": "ok", "timestamp": 1630019023457, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f1773067-31b3-41b4-db9f-b21a6d5dbf34"
data.rename(columns={'Unnamed: 0': 'movie_id'}, inplace=True)
columns=['movie_id','Cast','Director','Genre','Movie Name','Description']
data = data[columns]
data.isnull().values.any()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} id="WFcmwUFmsSfB" executionInfo={"status": "ok", "timestamp": 1630019023459, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aaa0b5ad-ae02-4263-98ad-5f7351f08b12"
def get_important_features(data):
    important_features=[]
    for i in range (0,data.shape[0]):
        important_features.append(data['Movie Name'][i]+' '+data['Director'][i]+' '+data['Genre'][i]+' '+data['Description'][i])
    return important_features

#creating a column to hold the combined strings
data['important_features'] = get_important_features(data)
data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="PFaRcjqXsw-G" executionInfo={"status": "ok", "timestamp": 1630019028049, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9d686d3a-75b8-4a3e-90b5-cdb0cfb1dcb4"
tfidf = TfidfVectorizer(stop_words='english')
#data['Description'] = data['Description'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['important_features'])
tfidf_matrix.shape
```

```python id="TrUd15_ss0GK" executionInfo={"status": "ok", "timestamp": 1630019028859, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index=data['Movie Name']).drop_duplicates()
#indices['Stillwater']
#sim_scores = list(enumerate(cosine_sim[indices['Stillwater']]))
```

```python id="W24iL9Fhs8R4" executionInfo={"status": "ok", "timestamp": 1630019029543, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 5 most similar movies
    movies=data['Movie Name'].iloc[movie_indices]
    id=data['movie_id'].iloc[movie_indices]
    dict={"Movies":movies,"id":id}
    final_df=pd.DataFrame(dict)
    final_df.reset_index(drop=True,inplace=True)
    return final_df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="-bYfnUiIs_WT" executionInfo={"status": "ok", "timestamp": 1630019030163, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4d57a25e-9ec5-4818-bea7-83e05c899b09"
get_recommendations('Spider-Man: Far from Home')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="QTDi3hDrtAnF" executionInfo={"status": "ok", "timestamp": 1630019037275, "user_tz": -330, "elapsed": 724, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="87386069-820d-4e2b-ccda-3375496beaec"
get_recommendations('Stillwater')
```

```python id="67x4JXXDrOMS" executionInfo={"status": "ok", "timestamp": 1630019177624, "user_tz": -330, "elapsed": 3114, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
pickle.dump(data, open('./artifacts/imdb_movie_list.pkl','wb'))
pickle.dump(cosine_sim, open('./artifacts/imdb_cosine_similarity.pkl','wb'))
```

<!-- #region id="oICbvFPTtl4f" -->
## Creating a frontend streamlit app for model serving
<!-- #endregion -->

```python id="_EzPxs2zu9l8" executionInfo={"status": "ok", "timestamp": 1630019519735, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir -p ./apps
```

```python colab={"base_uri": "https://localhost:8080/"} id="otbXLcBntvDd" executionInfo={"status": "ok", "timestamp": 1630019546388, "user_tz": -330, "elapsed": 630, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c1599731-f11c-4260-de32-af1e0ac85f83"
%%writefile ./apps/imdb_streamlit.py
import pickle
import streamlit as st

def recommend(movie):
    index = movies[movies['Movie Name'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    for i in distances[1:6]:
        recommended_movie_names.append(movies.iloc[i[0]]['Movie Name'])

    return recommended_movie_names

page_bg_img = '''
<style>
.stApp {
  background-image: url("https://payload.cargocollective.com/1/11/367710/13568488/MOVIECLASSICSerikweb_2500_800.jpg");
  background-size: cover;
}
</style>
'''

# st.markdown(page_bg_img, unsafe_allow_html=True)
# st.markdown(unsafe_allow_html=True)


st.header('Movie Recommendation System')
movies = pickle.load(open('./artifacts/imdb_movie_list.pkl','rb'))
similarity = pickle.load(open('./artifacts/imdb_cosine_similarity.pkl','rb'))

movie_list = movies['Movie Name'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)


if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie)
    for i in recommended_movie_names:
        st.write(i)
```

<!-- #region id="clc4K2TIw1bp" -->
Local testing
<!-- #endregion -->

```python id="AJ4EkzVGvoId"
!pip install -q streamlit
!pip install -q colab-everything
```

```python id="jpMhhQ7PvkPh" executionInfo={"status": "ok", "timestamp": 1630019706726, "user_tz": -330, "elapsed": 1910, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import streamlit as st
from colab_everything import ColabStreamlit
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="lCSgNM1tvI0R" executionInfo={"status": "error", "timestamp": 1630019989121, "user_tz": -330, "elapsed": 278401, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c5503baf-1d86-4231-b725-2c358f6c9436"
ColabStreamlit('./apps/imdb_streamlit.py')
```

<!-- #region id="EnoUKmVpwvkd" -->
<!-- #endregion -->

<!-- #region id="PhdjDsYkvuai" -->
## Packaging
<!-- #endregion -->

<!-- #region id="fYf7eNnpxI9A" -->
Dockerization
<!-- #endregion -->

```python id="9YWo4vz9xU2q" executionInfo={"status": "ok", "timestamp": 1630020493281, "user_tz": -330, "elapsed": 732, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir -p ./apps/imdb/v1_1/artifacts
!cp ./artifacts/imdb_cosine_similarity.pkl ./apps/imdb/v1_1/artifacts
!cp ./artifacts/imdb_movie_list.pkl ./apps/imdb/v1_1/artifacts
```

```python colab={"base_uri": "https://localhost:8080/"} id="epfyVmPzy8Ta" executionInfo={"status": "ok", "timestamp": 1630020580540, "user_tz": -330, "elapsed": 738, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9eb0c2c3-835d-4a7b-d967-502f3711595e"
%%writefile ./apps/imdb/v1_1/requirements.txt
Jinja2==2.11.3
jinja2-time==0.2.0
joblib==1.0.1
matplotlib-inline==0.1.2
numpy==1.19.1
numpydoc==1.1.0
pandas==1.1.3
pickleshare==0.7.5
Pillow==8.3.1
pip==21.0.1
ptyprocess==0.7.0
pyaml==19.4.1
PyYAML==5.4.1
requests==2.25.1
requests-aws4auth==0.9
scikit-learn==0.24.2
scipy==1.7.1
streamlit==0.81.1
urllib3==1.26.6
xlrd==1.2.0
yapf==0.31.0
zipp==3.5.0
```

```python colab={"base_uri": "https://localhost:8080/"} id="mKq2OR5fxmhU" executionInfo={"status": "ok", "timestamp": 1630020502808, "user_tz": -330, "elapsed": 917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bcb25402-7f03-43bc-c12c-dc61c4dc9934"
%%writefile ./apps/imdb/v1_1/app.py
import pickle
import streamlit as st

def recommend(movie):
    index = movies[movies['Movie Name'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    for i in distances[1:6]:
        recommended_movie_names.append(movies.iloc[i[0]]['Movie Name'])

    return recommended_movie_names


st.header('Movie Recommendation System')
movies = pickle.load(open('./artifacts/imdb_movie_list.pkl','rb'))
similarity = pickle.load(open('./artifacts/imdb_cosine_similarity.pkl','rb'))

movie_list = movies['Movie Name'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)


if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie)
    for i in recommended_movie_names:
        st.write(i)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xJjPYBUcxL2z" executionInfo={"status": "ok", "timestamp": 1630020639101, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a669c0ee-263b-4777-84ad-49f99b07ef97"
%%writefile ./apps/imdb/v1_1/Dockerfile
  
# lightweight python
FROM python:3.7-slim

RUN apt-get update

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN ls -la $APP_HOME/

# Install dependencies
RUN pip install -r requirements.txt

# Run the streamlit on container startup
CMD [ "streamlit", "run","--server.enableCORS","false","app.py" ]
```

```python colab={"base_uri": "https://localhost:8080/"} id="5BrbgQizzFRz" executionInfo={"status": "ok", "timestamp": 1630020757578, "user_tz": -330, "elapsed": 1028, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="34bb4b0d-0ec2-4b97-a76f-0e8f4bcb6111"
%%writefile ./apps/imdb/v1_1/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: movierecommendation
spec:
  replicas: 1
  selector:
    matchLabels:
      app: movie
  template:
    metadata:
      labels:
        app: movie
    spec:
      containers:
      - name: mveimdb
        image: sparshai/mveimdb
        ports:
        - containerPort: 8501
```

```python colab={"base_uri": "https://localhost:8080/"} id="onJa-7khzuAR" executionInfo={"status": "ok", "timestamp": 1630020787245, "user_tz": -330, "elapsed": 778, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="90dfc039-51ec-4fe9-8319-11454facb930"
%%writefile ./apps/imdb/v1_1/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: movie
spec:
  type: LoadBalancer
  selector:
    app: movie
  ports:
  - port: 80
    targetPort: 8501
```

<!-- #region id="j1cMjK9j11Yc" -->
## Deployment
<!-- #endregion -->

<!-- #region id="PVaQkEo212nj" -->
Commands
<!-- #endregion -->

```python id="QSi2Iyo8146G"
!docker build -t mveimdb .
!docker tag mveimdb:latest sparshai/mveimdb:latest
!docker push sparshai/mveimdb

!kubectl apply -f deployment.yaml
!kubectl apply -f service.yaml

!kubectl describe pods
!kubectl get services
```
