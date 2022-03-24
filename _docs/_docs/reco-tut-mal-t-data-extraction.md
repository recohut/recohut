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

```python id="jVkAVV4pixpb" executionInfo={"status": "ok", "timestamp": 1629179463408, "user_tz": -330, "elapsed": 1138, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mal"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="RDSfrKdHi4C8" executionInfo={"status": "ok", "timestamp": 1629179466786, "user_tz": -330, "elapsed": 2791, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="43522c86-71f9-4f9d-990c-efa4f6169878"
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

```python id="22P-ZOjbi4C_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629180306723, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="df49eaa1-b340-4777-c6c5-399b29a61edb"
!git status
```

```python id="9LDKaBYRi4DA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629180317294, "user_tz": -330, "elapsed": 838, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f3e68f9-79a5-49a7-89f5-9505a2d8447b"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="0cMqCJXmltX9" -->
---
<!-- #endregion -->

```python id="5iqk_gJnmDZp" executionInfo={"status": "ok", "timestamp": 1629179379601, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from bs4 import BeautifulSoup as bs
import requests
from IPython.display import clear_output
import pandas as pd
import numpy as np
import csv
```

<!-- #region id="uikBmz8JpCTA" -->
## Scraping the List
<!-- #endregion -->

<!-- #region id="Kytxt5b0mDZt" -->
### Preparing the recipe
<!-- #endregion -->

```python id="9fem73Z9mDZv" executionInfo={"status": "ok", "timestamp": 1629178468704, "user_tz": -330, "elapsed": 1309, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#Loading the webpage with anime list using 'request' library 
r = requests.get("https://myanimelist.net/topanime.php")

#reading the content with bautiful soup library
soup = bs(r.content)
contents = soup.prettify()

#find the table with anime titles in html sript and narrowing down elements to scrape
table = soup.find(class_="top-ranking-table")
title_rows = table.find_all("h3")
```

```python id="sBcs1jEZmDZ0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629178507299, "user_tz": -330, "elapsed": 493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="16fdaead-eb68-44e3-f57d-85235b32a120"
print(title_rows[0].prettify())
print(title_rows[1].prettify())
```

```python id="EHjNb3DTmDZ1" executionInfo={"status": "ok", "timestamp": 1629178623071, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#storing all titles and the web page links for the titles of first page
anime_list=[]
for row in title_rows:
    anime_link=[]
    anime_link.append(row.find("a").get_text())
    anime_link.append(row.a['href'])
    anime_list.append(anime_link)
```

```python colab={"base_uri": "https://localhost:8080/"} id="mTi4HbyrnPk-" executionInfo={"status": "ok", "timestamp": 1629178630464, "user_tz": -330, "elapsed": 467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9b7d5f28-ea9f-4773-abbb-a81186f813ab"
anime_list[0]
```

<!-- #region id="G9WmkwoFpIKa" -->
### Full scraping
<!-- #endregion -->

```python id="37-cnaY2mDZ9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629179344279, "user_tz": -330, "elapsed": 553808, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dc8304e3-6492-4c77-e11c-01453dc9ccb2"
anime_list1=[]
#each webpage has 50 titles and total 18349 titles were present running the loop to scrape info of all anime titles
for i in range(0,18301,50): 
    r1=requests.get("https://myanimelist.net/topanime.php?limit="+str(i))
    table1=(bs(r1.content)).find_all("h3")
    clear_output(wait=True)
    print("current progress:",i)
    for row in table1:
        anime_link=[]
        anime_link.append(row.find("a").get_text())
        anime_link.append(row.a['href'])
        anime_list1.append(anime_link)
```

```python id="Qcayg_WZmDaC" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629179354274, "user_tz": -330, "elapsed": 459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4445584f-38ac-4c35-e973-15e909814b98"
len(anime_list1)
```

```python id="ZJDzSfd_mDaJ" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1629179384052, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d7bdf643-cc73-4c25-d06e-ba0e29d6c6e6"
df = pd.DataFrame(anime_list1)
df.head()
```

```python id="reBdrELLmDaM" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1629179440493, "user_tz": -330, "elapsed": 599, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cc7b50c7-0b22-4cbf-874e-eafef795561a"
index_names = df[df[0]=="More"].index
df.drop(index_names,inplace=True)
df.reset_index(drop=True, inplace=True)
df.head()
```

```python id="z-Lt3aS9mDaO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629179447416, "user_tz": -330, "elapsed": 447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aeb3e262-7402-4f0d-d67c-a951e040cea9"
df.info()
```

```python id="yt41j-sXmDaN" executionInfo={"status": "ok", "timestamp": 1629179953988, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df.to_csv('./data/bronze/anime_list.csv', index=None, header=False)
```

<!-- #region id="joFHsQHFpLfv" -->
## Scraping the Info
<!-- #endregion -->

<!-- #region id="zr5m6Cg3pVtw" -->
### Preparing the recipe
<!-- #endregion -->

```python id="P1gbsjrZpMpZ" executionInfo={"status": "ok", "timestamp": 1629179717487, "user_tz": -330, "elapsed": 1486, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#loading the web page with request library
r = requests.get("https://myanimelist.net/anime/5114/Fullmetal_Alchemist__Brotherhood")
soup = bs(r.content)

#all the information required was there in class=borderClass (found with inspect element in web page)
AnimeDetails = soup.find(class_="borderClass")
```

```python id="8o5d_uP0pvgs" executionInfo={"status": "ok", "timestamp": 1629179722944, "user_tz": -330, "elapsed": 487, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#all required informaiton was under different 'div' elements
AnimeInfo=[]
for index,i in enumerate(AnimeDetails.findAll('div')):
    if index>=6:
        content_value=i.get_text()
        AnimeInfo.append(content_value)

#cleaning the data and removing unwanted formats and elements 
details=[]
for i in range(0 , len(AnimeInfo)):
    AnimeInfo[i]= AnimeInfo[i].replace('\n','')
    details.append(AnimeInfo[i].split(':',1))
details.append(['Title','Fullmetal Alchemist: Brotherhood'])
```

```python id="PNbWFCegpxdJ" executionInfo={"status": "ok", "timestamp": 1629179959180, "user_tz": -330, "elapsed": 706, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#getting the info of all the titles and web page links that were scrapped in last file
with open('./data/bronze/anime_list.csv', newline='', encoding="utf8") as f:
    reader = csv.reader(f)
    data = list(reader)
```

<!-- #region id="NB6XYChdrqnt" -->
### Full scraping
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="H8poD9y7sIBz" executionInfo={"status": "ok", "timestamp": 1629179964425, "user_tz": -330, "elapsed": 535, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="40127336-1698-4433-a81c-cfc8a0308125"
data[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="NBLwpTDar0f5" executionInfo={"status": "ok", "timestamp": 1629180208476, "user_tz": -330, "elapsed": 28905, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0b67e193-fbaa-4bca-fb49-36ec13612fe2"
#total 18,349 titles were present in the website
anime = pd.DataFrame()
for j in range(0,18349):
    df1=pd.DataFrame()
    url=data[j][1]
    r=requests.get(url)
    soup=bs(r.content)
    AnimeDetails=soup.find(class_="borderClass")
    AnimeInfo=[]
    if j%10==0:
        clear_output(wait=True)
        print("current progress:",j)
    for index,i in enumerate(AnimeDetails.findAll('div')):
        if index>=6:
            content_value=i.get_text()
            AnimeInfo.append(content_value)
    details=[]
    for k in range(0 , len(AnimeInfo)):
        AnimeInfo[k]= AnimeInfo[k].replace('\n','')
        details.append(AnimeInfo[k].split(':',1))
    title=[]
    title.append("Title")
    title.append(data[j][0])
    details.append(title)
    df1=pd.DataFrame(details)
    df1=df1.set_index(0)
    df1=df1.transpose()
    df1.dropna(axis=1,inplace=True)
    frames=[anime,df1]
    anime=pd.concat(frames)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="D2K49AUftQKj" executionInfo={"status": "ok", "timestamp": 1629180728465, "user_tz": -330, "elapsed": 828, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bc58ab0f-1287-444f-9714-05db9688abe7"
anime.head(10)
```

```python id="xpwQu1eJsCp7"
# Saving the data as csv file
anime.to_csv('./data/bronze/anime_data.csv', compression='gzip')
```
