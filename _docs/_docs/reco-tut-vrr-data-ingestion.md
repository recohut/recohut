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

```python id="QHfJAkdQk3jv"
import os
project_name = "reco-tut-vrr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3APW7y3clBlH" outputId="e8a9f0ea-a594-4377-fb7f-65d6e116c420"
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

```python colab={"base_uri": "https://localhost:8080/"} id="yGo_LJd6lBlN" outputId="68558c87-5331-414d-bee3-115c6cf08e7e"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="SWzR_TDJlBlO" outputId="1f0aa5a6-f8e0-4bd1-8b83-cea3865fa6dc"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="ra084G6QmDcL" -->
---
<!-- #endregion -->

```python id="Nrb7EyBKmEmk"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import cufflinks as cf
```

```python colab={"base_uri": "https://localhost:8080/"} id="ce393Lpposh_" outputId="514ef85a-ec60-4741-d7e0-cf47b66557e1"
nltk.download('wordnet')
nltk.download('stopwords')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="HF4J7q62mJ7u" outputId="fa589785-67ca-4d19-d153-ff88baef9716"
df = pd.read_csv('./data/bronze/reviews.csv', na_values=' ')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="knyLYsCEmT0d" outputId="44f51efb-3929-4f88-a44b-903e71fee2f7"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="PigPFFn7mU9B" outputId="77821be0-4273-4311-f9e1-d78d3d271617"
print('Unique counts:', df.nunique())
```

```python colab={"base_uri": "https://localhost:8080/"} id="JF80KYZrmZfX" outputId="2e3355ac-7c69-45fa-c35d-75b1af2dc09b"
print('Kind of ratings:',df.Rating.unique())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="_6URsZ09miaT" outputId="6e1ce849-f94e-494a-9887-943a2d9465e7"
#remove extra columns and keep the necessary ones for analysis
df = pd.DataFrame(df.drop(['Bubble_Count', 'Review_Count'], axis=1))

#remove new line characters from variable columns
df = df.replace(r'\n',' ', regex=True)

#remove the numbers in the review column
df.Reviews = df.Reviews.str.replace('\d+', '')

#fills the rating and Review_counts variable missing values with the mean and median respectively
df = df.fillna({'Rating': df.Rating.median(), 'Review_counts': df.Review_counts.mean()})

#drop all missing values
df.dropna(axis=0, how='any', inplace=True)

#reset index
df.reset_index(drop=True, inplace=True)

df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="XgOz-tESnBTx" outputId="d9c0b9d8-d9ae-4f11-e722-397345f832ab"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="TiE55fy8oFVt" outputId="8af9576d-ba9a-4653-fb2d-6b030b0324aa"
# Let's understand the two lists: reviews (text_train) and their labels (y_train)
print("Type : ", type(df.Reviews))
print("Length of reviews: ", len(df.Reviews))
print("Review at index 6:\n ", df.Reviews[6])
print("Label of the review at Index 6: ", df.Rating[6])
```

<!-- #region id="TTF9WlMwyE-7" -->
Vacation Rentals Reviews Word Count Distribution
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nHluRdemyHSb" outputId="689fc356-9ae0-4225-d6c2-8a341c0ca863"
df['word_count'] = df['Reviews'].apply(lambda x: len(str(x).split()))
reviews_lengths = list(df['word_count'])
print("Number of descriptions:",len(reviews_lengths),
      "\nAverage word count", np.average(reviews_lengths),
      "\nMinimum word count", min(reviews_lengths),
      "\nMaximum word count", max(reviews_lengths))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="Nd4ZPGNWx5Q4" outputId="a007bbb2-fbdd-4328-9d10-7307305f0ad9"
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
df['word_count'].iplot(
    kind='hist',
    bins = 50,
    linecolor='black',
    xTitle='word count',
    yTitle='count',
    title='Word Count Distribution in rental reviews')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="xF7dSfJ_qUq5" outputId="45f274f0-ff6b-4b34-961f-0d638a855bb7"
#Shows the average ratings of each city and province and the number of reviews obtained from each city
#The higher number of ratings indicates what cities are getting more visits and in which province visits are more
df.groupby(['Province', 'City']).agg({'Rating':'mean','Review_counts':'sum'}).sort_values(by= ['Province','Review_counts'], ascending=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="0pAJhiUpqUkW" outputId="7f61a8c8-3869-4bb5-9e20-e8e6f37add6a"
#Showing the best performing cities in terms of Review Ratings
a = df.groupby(['City']).agg({'Rating':'mean','Review_counts':'sum'}).sort_values(by= ['Rating'], ascending=False)
b = a[:10] #the 10 best performing cities
b
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="_dj-HLhyqmOn" outputId="59166625-4920-43af-9487-669d046df4cf"
z = a.tail(10)  #the 10 lowest performing cities
z
```

```python colab={"base_uri": "https://localhost:8080/"} id="0009ybBanWUM" outputId="ae9dd4e9-b5a7-4e18-d01a-5cf4533ab0f1"
#Label Encoding for Categorical Target Variable
lb = LabelEncoder()
df['Rating'] = lb.fit_transform(df['Rating'])
y = df.Rating
Rating = df.Rating
df.Rating.value_counts()
```

```python id="5fKU9Cw8nDHi"
!mkdir -p ./data/silver

df.to_parquet('./data/silver/reviews.parquet.gzip', compression='gzip')
```
