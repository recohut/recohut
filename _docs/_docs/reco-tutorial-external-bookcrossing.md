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

<!-- #region id="O25aIENKWPZg" -->
# <font color = 'blue'><b> Book Recommender System

<!-- #endregion -->

<!-- #region id="24zRjUtSXNnI" -->
<font color = 'blue'> During the last few decades, with the rise of Youtube, Amazon, Netflix, and many other such
web services, recommender systems have taken more and more place in our lives. From
e-commerce (suggest to buyers articles that could interest them) to online advertisement
(suggest to users the right contents, matching their preferences), recommender systems are
today unavoidable in our daily online journeys.
In a very general way, recommender systems are algorithms aimed at suggesting relevant
items to users (items being movies to watch, text to read, products to buy, or anything else
depending on industries).

<font color = 'blue'> Recommender systems are really critical in some industries as they can generate a huge
amount of income when they are efficient or also be a way to stand out significantly from
competitors. The main objective is to create a book recommendation system for users.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nGUn89kZGcHp" outputId="1c5841c2-182f-4480-b1dc-2bab6eedb3ef"
#mounting drive
from google.colab import drive
drive.mount('/content/drive')
```

```python id="58aqrotgE3HG"
#importing libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
```

```python id="xfDFTdzmaryk"
import seaborn as sns
```

<!-- #region id="6mjV1Gh7Y4OI" -->
# <font color = 'blue'><b> About Dataset
<!-- #endregion -->

<!-- #region id="U4EDXZT_ZB1d" -->
<font color = 'blue'> The Book-Crossing dataset comprises 3 files.

<font color = 'blue'>● Users

<font color = 'blue'>Contains the users. Note that user IDs (User-ID) have been anonymized and map to
integers. Demographic data is provided (Location, Age) if available. Otherwise, these
fields contain NULL values.

<font color = 'blue'>● Books

<font color = 'blue'>Books are identified by their respective ISBN. Invalid ISBNs have already been removed
from the dataset. Moreover, some content-based information is given (Book-Title,
Book-Author, Year-Of-Publication, Publisher), obtained from Amazon Web
Services. Note that in the case of several authors, only the first is provided. URLs linking
to cover images are also given, appearing in three different flavors (Image-URL-S,
Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the
Amazon website.

<font color = 'blue'>● Ratings


<font color = 'blue'>Contains the book rating information. Ratings (Book-Rating) are either explicit,
expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ovcb53aLGba5" outputId="f48c9e7c-efb3-4b0e-d500-98e2e2dbe5bc"
# Read the books.csv data
pathBooks = "/content/drive/MyDrive/AlmaBetter/Capstone_Project/Capstone III/data_book_recommendation/Books.csv"
dfbooks = pd.read_csv(pathBooks)
#df1 = pd.DataFrame(df)
```

```python id="lTZISxRZTr0a"
# Read the Ratings.csv data
pathRatings = "/content/drive/MyDrive/AlmaBetter/Capstone_Project/Capstone III/data_book_recommendation/Ratings.csv"
dfratings = pd.read_csv(pathRatings)
#df1 = pd.DataFrame(df)
```

```python id="GZHwaXN8UN55"
# Read the Users.csv data
pathUsers = "/content/drive/MyDrive/AlmaBetter/Capstone_Project/Capstone III/data_book_recommendation/Users.csv"
dfusers = pd.read_csv(pathUsers)
#df1 = pd.DataFrame(df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 428} id="oxN5Xif3UWmw" outputId="5d1509fe-ef4b-414f-9853-2f8f95f3e3ea"
dfbooks.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 496} id="Qu1aekzMUwNQ" outputId="6e0d679b-3803-4a01-d9b8-ebe576ae6be8"
dfbooks.tail()
```

```python colab={"base_uri": "https://localhost:8080/"} id="xGJwrbXdUZMa" outputId="6aeb7200-c451-4588-9287-679e5c8fe6ed"
dfbooks.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="MoVdhdIeUbwT" outputId="392a83ea-847a-4967-d482-2c78df13da76"
dfbooks.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ArMOOEPjUeim" outputId="3a793e5b-7518-48d7-cbee-8687033f0150"
dfbooks.isnull().sum()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 369} id="HEsOfHi9FfUs" outputId="23608513-d581-41d4-915a-3f4264bfd782"
dfbooks.loc[dfbooks['Year-Of-Publication'] == 'DK Publishing Inc', :]
```

```python id="cTgQb3okFtCZ"
dfbooks.loc[dfbooks.ISBN == '0789466953','Year-Of-Publication'] = 2000
dfbooks.loc[dfbooks.ISBN == '0789466953','Book-Author'] = "James Buckley"
dfbooks.loc[dfbooks.ISBN == '0789466953','Publisher'] = "DK Publishing Inc"
dfbooks.loc[dfbooks.ISBN == '0789466953','Book-Title'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)\";James Buckley"
```

```python id="9oeNBZjlGar9"
dfbooks.loc[dfbooks.ISBN == '078946697X','Year-Of-Publication'] = 2000
dfbooks.loc[dfbooks.ISBN == '078946697X','Book-Author'] = "JMichael Teitelbaum"
dfbooks.loc[dfbooks.ISBN == '078946697X','Publisher'] = "DK Publishing Inc"
dfbooks.loc[dfbooks.ISBN == '078946697X','Book-Title'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)\";Michael Teitelbaum"
```

```python colab={"base_uri": "https://localhost:8080/", "height": 168} id="OUTEPCSfGxWp" outputId="e8208162-0159-4688-dbfc-09ba0c83481e"
dfbooks.loc[dfbooks['Year-Of-Publication'] == 'Gallimard', :]
```

```python id="KaJR4jY8G5Mw"
dfbooks.loc[dfbooks.ISBN == '2070426769','Year-Of-Publication'] = 2003
dfbooks.loc[dfbooks.ISBN == '2070426769','Book-Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
dfbooks.loc[dfbooks.ISBN == '2070426769','Publisher'] = "Gallimard"
dfbooks.loc[dfbooks.ISBN == '2070426769','Book-Title'] = "Peuple du ciel, suivi de Les Bergers"
```

```python id="gqo3vM16HKVB"
dfbooks['Year-Of-Publication'] = pd.to_numeric(dfbooks['Year-Of-Publication'], errors = 'coerce')
```

```python colab={"base_uri": "https://localhost:8080/"} id="_Q06jdv7HV1F" outputId="6073287e-075f-4fe1-eac2-1502ada3930f"
print(sorted(dfbooks['Year-Of-Publication'].unique()))

```

```python id="2BwUvNwtJ5oU"
#sns.distplot(dfbooks['Year-Of-Publication'], kde=False, hist_kws={"range": [1945,2020]})
```

```python colab={"base_uri": "https://localhost:8080/", "height": 306} id="SRmz712RKzYu" outputId="a6c224eb-c0e6-48a7-f0e5-90f0fc1fe8ed"
sns.countplot(x="Year-Of-Publication", data=dfbooks)
```

```python id="Xt-J9W2uHoRQ"
dfbooks.loc[(dfbooks['Year-Of-Publication'] > 2006) | (dfbooks['Year-Of-Publication'] == 0), 'Year-Of-Publication'] = np.NAN
dfbooks['Year-Of-Publication'].fillna(round(dfbooks['Year-Of-Publication'].median()), inplace = True)
```

```python id="-MWoJ32RHoDo"
dfbooks['Year-Of-Publication'] = dfbooks['Year-Of-Publication'].astype(np.int32)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 360} id="CNGVidoq3ViM" outputId="a755437f-d38f-4ebd-9559-879d7ba05f46"
# Publication by Year
#year = pd.to_numeric(dfbooks['Year-Of-Publication'], 'coerse').fillna(2099, downcast = 'infer')
sns.distplot(dfbooks['Year-Of-Publication'], kde=False, hist_kws={"range": [1945,2020]})
```

```python colab={"base_uri": "https://localhost:8080/", "height": 182} id="3iW4_xTjHn0U" outputId="b0554926-5a4b-491d-d46f-af1397d34185"
dfbooks.loc[dfbooks.Publisher.isnull(),:]
```

```python id="IO4BJGh2HnY4"
dfbooks.loc[(dfbooks.ISBN == '193169656X'), 'Publisher'] = 'other'
dfbooks.loc[(dfbooks.ISBN == '1931696993'), 'Publisher'] = 'other'
```

```python id="T3rVkt70gKSJ"
dfbooks.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
```

```python id="nUdPnP2rIeWV"
dfbooks.loc[dfbooks.ISBN == '9627982032','Book-Author'] = "David Tait"
```

```python id="juePRgvSIeKC"
dfbooks.loc[dfbooks.ISBN == '193169656X','Publisher'] = "Novelbooks Inc"
dfbooks.loc[dfbooks.ISBN == '1931696993','Publisher'] = "Bantam"
```

```python colab={"base_uri": "https://localhost:8080/"} id="CSmAS7bLId8H" outputId="549a067d-32e6-49cf-d497-0a30ce1f6db8"
dfbooks.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="b1OLeGi-Lcuj" outputId="b5259400-0933-400b-9683-2fd66d1c871f"
dfbooks.info()
```

```python id="Lq14fEQNLckg"

```

```python id="WA5i3cn6Lcbi"

```

```python id="E7srLkPBIdp8"

```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="11qn0GE7UqFA" outputId="e95c34d6-3429-454d-b915-ec3f780f88dd"
dfratings.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="3i--OplIVyyn" outputId="0d1f202e-c3f8-4614-b92f-c6f774455399"
dfratings.tail()
```

```python colab={"base_uri": "https://localhost:8080/"} id="hKJDSAQVV1B-" outputId="dd91fee6-d0ff-4dfd-d0ba-e6c61f200285"
dfratings.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="8A_gIgXDV2q2" outputId="2e4c850f-9302-40b5-b258-781dce7a70ff"
dfratings.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="1-PL1bRNV4ty" outputId="8497185f-aba0-4f3e-9d57-c7c375d07484"
dfratings.duplicated().sum()
```

<!-- #region id="NGXzV6MYYH9y" -->
<font color = 'blue'> We do not have any null values for the ratings data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 334} id="qVSN_F7ERzcs" outputId="9663e230-3c4b-4e16-8865-5dfde720d402"
plt.rc("font", size=15)
dfratings['Book-Rating'].value_counts(sort=False).plot(kind='bar')
plt.title('Ratings distribution \n')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()
```

```python id="OfLbM5jeNlVW"
unique_ratings = dfratings[dfratings.ISBN.isin(dfbooks.ISBN)]
```

```python id="CzgYky99MKRC"
ratings_explicit= unique_ratings[unique_ratings['Book-Rating'] != 0]
ratings_implicit= unique_ratings[unique_ratings['Book-Rating'] == 0]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="fXlcmv5aMKGu" outputId="7b107559-cc28-41b0-9d50-c82157dda436"
sns.set_style('darkgrid')
sns.countplot(data= ratings_explicit , x='Book-Rating', palette="Blues_d")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="NDO5jVSfPG_H" outputId="4e83ec87-7c25-46b7-8820-1c11c9b28052"
dfratings.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="z_Dtiob2MJ9G" outputId="3b08e231-726c-4bfb-ca22-10d9876247bd"
unique_ratings.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="hIy_Zum1MJxU" outputId="8a5d24dd-b036-4a8c-cc5d-8cb124b126da"
ratings_implicit.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="5CF-7sSmMJjh" outputId="8b15b7f9-fabc-4681-89b1-27f5ba1aad68"
ratings_explicit.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="zZbjeSSbjlSq" outputId="d8dc40e8-67d4-4964-cf9f-c2d61638b85e"
dfratings_count = pd.DataFrame(ratings_explicit.groupby('ISBN')['Book-Rating'].count())
dfratings_count.sort_values('Book-Rating', ascending = False).head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="dWc8ySIKmDJT" outputId="ef2da68c-bc16-4e7b-9f59-3211c8457758"
avg_rating= pd.DataFrame(ratings_explicit.groupby('ISBN')['Book-Rating'].mean())
avg_rating['ratingCount'] = pd.DataFrame(ratings_explicit.groupby('ISBN')['Book-Rating'].count())
avg_rating.sort_values('ratingCount', ascending= False).head()
```

```python id="6EALIOx9pO-5"
user_rating_count= pd.DataFrame(ratings_explicit['User-ID'].value_counts())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="xssmI0EtPbSN" outputId="d638751c-43c9-4bbe-acb9-5955af00f265"
user_rating_count
```

```python id="-8oXjCFZPbGQ"

```

```python id="RDNT6rGlPa6N"

```

<!-- #region id="cjWKcgfqRRm5" -->
# <font color = 'blue'> Users data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="WA3WRQxOV7d3" outputId="bf7b4033-8237-421d-f5c3-4e0d242186ac"
dfusers.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="7RnpYmAGWCuj" outputId="32a75fef-c7d4-466f-fa84-a498f5ab5302"
dfusers.tail()
```

```python colab={"base_uri": "https://localhost:8080/"} id="gP910vTJWEOq" outputId="97635e8f-4841-4e2f-a5cc-2b4fc05e29f3"
dfusers.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="klfKrS1MWF5F" outputId="5666bc0a-9fe1-4761-e18e-f39ae6257325"
dfusers.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="-OnOEQSjWH46" outputId="20a89e39-43ff-46da-e6ba-dfa2790814c0"
dfusers.isnull().sum()
```

```python colab={"base_uri": "https://localhost:8080/"} id="CqxBJY7rDZgW" outputId="70bf06d5-f3a0-49e3-ce92-875008337330"
dfusers.nunique()
```

<!-- #region id="WEkyLCORYXWM" -->
<font color = 'blue'> We have 110762 null values in Age column for Users data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8RghWqywj2LK" outputId="a797e265-ae32-49b1-f148-7e630825a7f7"
print(sorted(dfusers.Age.unique()))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 330} id="1Uqdlh1If7A8" outputId="94f3ddee-6dec-4277-fe9b-ddf5cd9726d7"
dfusers.Age.hist(bins=[0,10,20,30,50,100])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 360} id="TaDeHDw_IIm1" outputId="c155758b-1762-433b-9307-a9e403d9de3a"
sns.distplot(dfusers['Age'].dropna())
```

```python id="baWLYXwEQp6R"
dfusers[['City','State','Country']] = dfusers.Location.str.split(",",expand=True,n=2)
```

```python id="z_9efIciRoj7"
dfusers.drop(['Location','City','State'],axis=1,inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Yl5hSFYBQp3b" outputId="766fbaa6-559b-4e34-bc8a-38de1da71344"
dfusers['Country'].unique()
```

```python colab={"base_uri": "https://localhost:8080/"} id="YJ3eaHJWQpzw" outputId="ade0122a-77d7-46c7-921b-f7f4d50da02a"
dfusers.nunique()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6Nqt5ZlQTEJE" outputId="bbac4bd8-a0c5-45e3-d697-f4d0d95951d2"
dfusers.isnull().sum()
```

```python id="sNZGeDjETEA0"
dfusers['Country']=dfusers['Country'].astype('str')
```

```python id="NC12BNqkTDxO"
dfusers.loc[(dfusers.Age > 90) | (dfusers.Age < 5), 'Age'] = np.nan
```

```python id="hhOtKGiPTfvD"
dfusers = dfusers[dfusers['Country'].notna()]
```

```python id="irmVxCAfTfgs"
dfusers['Age'] = dfusers['Age'].fillna(dfusers.groupby('Country')['Age'].transform('median'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="SuUK83G0UZQ7" outputId="a5aca3f4-76a8-4db7-98a7-41f3be772a27"
dfusers.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="iSW4gBnHUZDL" outputId="7439df74-3350-4138-e125-beefba432985"
dfusers.isna().sum()
```

```python id="AjwmwPSMUY_J"
dfusers['Age'].fillna(dfusers.Age.mean(),inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="n2qvAVFoUkmg" outputId="a668e1de-b2c6-45c5-a652-2f5eb674d566"
dfusers.isna().sum()
```

```python id="JMz77LizVIhf"

```

```python id="b0IAuegvVIXs"

```

```python id="lvPDtQPBUYNQ"

```

```python colab={"base_uri": "https://localhost:8080/"} id="-L592_MbWKjS" outputId="877324c4-da18-4577-8052-5c5204626cc8"
len(dfbooks["Book-Title"].unique())
```

```python colab={"base_uri": "https://localhost:8080/"} id="4rDnK09uee9h" outputId="f27ed40d-83e1-4e30-8d02-0b6262def41a"
len(dfbooks["Book-Author"].unique())
```

```python colab={"base_uri": "https://localhost:8080/"} id="WFpDzZnueny2" outputId="38f90aad-fef2-46af-9c76-3b6d0da965a6"
len(dfbooks["Publisher"].unique())
```

```python colab={"base_uri": "https://localhost:8080/"} id="iyzwIfmSesDv" outputId="444b1af0-522b-415e-c80a-f45a97abe5a7"
len(dfratings["User-ID"].unique())
```

```python colab={"base_uri": "https://localhost:8080/"} id="anOt8JW4fsSU" outputId="d0edb8e5-a63f-4b2f-f3c4-6e6836e84b45"
len(dfratings["ISBN"].unique())
```

```python colab={"base_uri": "https://localhost:8080/"} id="C9t3VALIfwlY" outputId="0e9bd86c-3201-4f3d-e7ff-7d26992a1458"
len(dfratings["Book-Rating"].unique())
```

```python id="GEHPWTfqkHJa"

```

```python colab={"base_uri": "https://localhost:8080/"} id="aHanj0W9sfc4" outputId="8e293188-f8f4-495c-9aa9-e32aceb132a9"
dfratings['User-ID'].nunique()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Zd_9_o_2yqVE" outputId="712fee84-7caf-4d2b-f902-72d06e4602ff"
dfusers['User-ID'].nunique()
```

```python colab={"base_uri": "https://localhost:8080/"} id="PXjZjtcey8vV" outputId="abcc98ab-073c-4575-b1a6-beb9f2b870cb"
dfbooks['ISBN'].nunique()
```

```python id="j9OkJV_X7IXp"

```

```python id="iTKvmQL87IM5"

```

```python id="l0JxZCaN7H-4"

```

<!-- #region id="37XAlXD1WDRB" -->
#<font color = 'blue'> Popularity based
<!-- #endregion -->

<!-- #region id="WpuZ4ccFXq0N" -->
<font color = 'blue'> Most rated
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="h9gCoNPlWMXU" outputId="ec2ae054-4a84-4fec-e335-fe9c256c62ca"
ratings_count = pd.DataFrame(ratings_explicit.groupby(['ISBN'])['Book-Rating'].sum())
top10 = ratings_count.sort_values('Book-Rating', ascending = False).head(10)
print("Following books are recommended")
top10=top10.merge(dfbooks, left_index = True, right_on = 'ISBN')
top10['Book-Title']
```

```python id="7hQdQ3v3XzyS"

```

<!-- #region id="2WvrxO3ag8TU" -->
#CF 
<!-- #endregion -->

```python id="EaHL4Fq6WMCa"
counts1 = ratings_explicit['User-ID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['User-ID'].isin(counts1[counts1 >= 100].index)]
counts = ratings_explicit['Book-Rating'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['Book-Rating'].isin(counts[counts >= 100].index)]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 318} id="aZvJz9Rzac_0" outputId="2ee64dd3-d587-4eda-cac5-aede5257174e"
#Generating ratings matrix from explicit ratings table
ratings_matrix = ratings_explicit.pivot(index='User-ID', columns='ISBN', values='Book-Rating')
userID = ratings_matrix.index
ISBN = ratings_matrix.columns
print(ratings_matrix.shape)
ratings_matrix.head()
#Notice that most of the values are NaN (undefined) implying absence of ratings
```

```python colab={"base_uri": "https://localhost:8080/"} id="wruIENN1acvs" outputId="9af89b7f-5ff4-4f3c-955e-dc6eba0d4d81"

n_users = ratings_matrix.shape[0] #considering only those users who gave explicit ratings
n_books = ratings_matrix.shape[1]
print (n_users, n_books)
```

```python id="G3z3te7Ha16w"
ratings_matrix.fillna(0, inplace = True)
ratings_matrix = ratings_matrix.astype(np.int32)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 301} id="Ab1KVeRya4oe" outputId="cd0ae0ef-2aeb-4679-b318-0386ebfa52a2"
#checking first few rows
ratings_matrix.head(5)
```

```python id="oVQhWnHIbuSy"
users_exp_ratings = dfusers[dfusers['User-ID'].isin(ratings_explicit['User-ID'])]
users_imp_ratings = dfusers[dfusers['User-ID'].isin(ratings_implicit['User-ID'])]
```

```python id="LerWGtlta4eJ"
sparsity = 1.0-len(ratings_explicit)/float(users_exp_ratings.shape[0]*n_books)
```

```python id="8HVrdwwsbBU1"

```

```python id="y7i2XZ3cbBG6"

```

```python colab={"base_uri": "https://localhost:8080/"} id="d3k-cnOcbA5f" outputId="35b21418-3e78-430a-dd0b-2e4e40b39246"
users_interactions_count_df = ratings_explicit.groupby(['ISBN', 'User-ID']).size().groupby('User-ID').size()
print('# of users: %d' % len(users_interactions_count_df))

users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['User-ID']]
print('# of users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))
```

```python colab={"base_uri": "https://localhost:8080/"} id="QOW2vomahqBB" outputId="6ba04a81-8bfd-49b5-e1a5-e4c90e81d33e"
print('# of interactions: %d' % len(ratings_explicit))
interactions_from_selected_users_df = ratings_explicit.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'User-ID',
               right_on = 'User-ID')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="JdzUt566hpva" outputId="6af69b27-c96f-48b0-b390-5c5fb5e9b702"
interactions_from_selected_users_df.head(10)
```

```python id="xMRdT4pCimZ9"
import math
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="NnQ1Wa-ziD5F" outputId="e18638b5-9bb7-416e-b42c-8cd32ee18801"
def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df.groupby(['ISBN', 'User-ID'])['Book-Rating'].sum().apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head()
```

```python id="I_2u8WaPi8ai"
from sklearn.model_selection import train_test_split
```

```python colab={"base_uri": "https://localhost:8080/"} id="lj6TfyutiDrr" outputId="082f49c1-df81-4cae-f085-9f68d82d57cb"
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['User-ID'], 
                                   test_size=0.20,
                                   random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="mPvU4dImiyhp" outputId="398ab088-e0af-4eae-8906-d4520271652f"
interactions_test_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 301} id="PfERiDMUiyeQ" outputId="e06e82ef-d2a8-4c3d-ef82-a56ea64dd739"
#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='User-ID', 
                                                          columns='ISBN', 
                                                          values='Book-Rating').fillna(0)

users_items_pivot_matrix_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="zj1rz0htiya8" outputId="25d5cd84-ed98-491f-c6a7-5a240806c23a"
users_items_pivot_matrix = users_items_pivot_matrix_df.values
users_items_pivot_matrix[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="AvzoOQYwjSSa" outputId="9bcb17ee-ee87-4fe8-d49f-8aeabd843f69"
users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]
```

```python id="q2MO27nkjSGm"
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
```

```python id="aMi2VzX1jaXI"
# The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15

#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
```

```python colab={"base_uri": "https://localhost:8080/"} id="rmFH5JW8jR40" outputId="2cc06024-a255-4739-9694-7d36e5d2910f"
users_items_pivot_matrix.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="fombLP2kjf6A" outputId="a9cf84e4-48eb-4954-f968-a1e08563f0b0"
U.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ks-j0521jfo9" outputId="2e150e02-5ffc-4b4a-e391-48ab9519f3d0"
sigma = np.diag(sigma)
sigma.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="ivR-f8POkagL" outputId="369cde68-5633-441e-da1d-958f878f2bcf"
Vt.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="jLtgLehkkaTa" outputId="3bef4d6e-1841-487b-8db4-d1095f24dd26"
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings
```

```python colab={"base_uri": "https://localhost:8080/"} id="AC3HyC_DkaFE" outputId="746400c8-2abe-4a9f-f34a-50953471491e"
all_user_predicted_ratings.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 369} id="h-LFkpMzkoBc" outputId="00cee3e2-1e36-4118-da3f-644b3ee2ba97"
#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="CKTq57b8knsC" outputId="5a3d425f-891c-4158-b4b9-9b8e49942c57"
len(cf_preds_df.columns)
```

```python id="miKz2g4RknoS"
class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df):
        self.cf_predictions_df = cf_predictions_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating content that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['ISBN'].isin(items_to_ignore)].sort_values('recStrength', ascending = False).head(topn)

        return recommendations_df


cf_recommender_model = CFRecommender(cf_preds_df)
```

```python id="dWTc3dSdknhP"
#Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('User-ID')
interactions_train_indexed_df = interactions_train_df.set_index('User-ID')
interactions_test_indexed_df = interactions_test_df.set_index('User-ID')
```

```python id="8mNLdcQTknXS"
def get_items_interacted(UserID, interactions_df):
    interacted_items = interactions_df.loc[UserID]['ISBN']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])
```

```python id="rm3cs1cTknS3"
#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:

    # Function for getting the set of items which a user has not interacted with
    def get_not_interacted_items_sample(self, UserID, sample_size, seed=42):
        interacted_items = get_items_interacted(UserID, interactions_full_indexed_df)
        all_items = set(ratings_explicit['ISBN'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index
    
    # Function to evaluate the performance of model for each user
    def evaluate_model_for_user(self, model, person_id):
        
        # Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        
        if type(interacted_values_testset['ISBN']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['ISBN'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['ISBN'])])
            
        interacted_items_count_testset = len(person_interacted_items_testset) 

        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id, interactions_train_indexed_df),topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            
            # Getting a random sample of 100 items the user has not interacted with
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, seed=item_id)    #%(2**32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['ISBN'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['ISBN'].values
            
            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_recs_df,person_metrics       #person_recs_df extra

    
    # Function to evaluate the performance of model at overall level
    def evaluate_model(self, model):
        
        people_metrics = []
        
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):    
            person_recs_df, person_metrics = self.evaluate_model_for_user(model, person_id)      #person_recs_dfextra
            person_metrics['User-ID'] = person_id
            people_metrics.append(person_metrics)
            
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return person_recs_df, global_metrics, detailed_results_df    #person_recs_df extra
    
model_evaluator = ModelEvaluator()    
```

```python id="e__8INUNEqRe"
import random
```

```python colab={"base_uri": "https://localhost:8080/", "height": 444} id="3m9iI51mknPP" outputId="281117f8-c788-4c76-fdae-9110647ef3f1"
print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
person_recs_df, cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)

print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="L4jCxlsRJ-4V" outputId="6ddf922f-33ad-4422-f50c-9d1fa9650793"
#person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id, interactions_train_indexed_df),topn=10000000000)
print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
#person_recs_df, cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)

```

```python id="fSNg0oxmplgV" colab={"base_uri": "https://localhost:8080/"} outputId="f03ac95c-abe3-41f3-aea1-a3352012c8a7"
person_recs_df.shape
```

```python id="g2OeWGoIKxvX"
person_recs_df=pd.DataFrame(person_recs_df)
```

```python id="NdyYwTheKxeg"
person_recs_df=person_recs_df[0:10]
```

```python id="jY3Zc3zkau3d"
rec_books= person_recs_df.merge(dfbooks, how='inner', left_on = 'ISBN', right_on = 'ISBN')
```

```python id="kMPXNIf4plZG" colab={"base_uri": "https://localhost:8080/"} outputId="0ed95186-edb3-4c5a-9145-c9a3df9b4b56"
rec_books['Book-Title']
```

```python id="G9T4hPVBzPWk"
#ratingsPivot= dfratings.pivot(index = 'User-ID', columns= 'ISBN').Book-Rating
#UserId= ratingsPivot.index
#ISBN = ratingsPivot.columns
#print(ratingsPivot.shape)
#ratingsPivot.head()
```

```python id="Gqf1ML7846li" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="c01c1b0e-fc07-408a-c941-16ec333c501e"
new_book_df= pd.merge(dfbooks, ratings_explicit, on='ISBN')
new_book_df.head()
```

```python id="YwQsHa4B47CR"
from sklearn import model_selection
train_data, test_data = model_selection.train_test_split(new_book_df, test_size=0.20)
```

```python colab={"base_uri": "https://localhost:8080/"} id="_zUJHotAW4Bq" outputId="26a99129-8778-4785-9b22-0ffdcbe8d4ed"
print(f'Training set lengths: {len(train_data)}')
print(f'Testing set lengths: {len(test_data)}')
print(f'Test set is {(len(test_data)/(len(train_data)+len(test_data))*100):.0f}% of the full dataset.')
```

```python id="uzAZu5l5W-gl"
# Get int mapping for user_id in train dataset

u_unique_train = train_data['User-ID'].unique()  
train_data_user2idx = {o:i for i, o in enumerate(u_unique_train)}

# Get int mapping for isbn in train dataset

i_unique_train = train_data.ISBN.unique()  
train_data_book2idx = {o:i for i, o in enumerate(i_unique_train)}

# Get int mapping for user_id in test dataset

u_unique_test = test_data['User-ID'].unique()  
test_data_user2idx = {o:i for i, o in enumerate(u_unique_test)}

# Get int mapping for isbn in train dataset

i_unique_test = test_data.ISBN.unique() 
test_data_book2idx = {o:i for i, o in enumerate(i_unique_test)}
```

```python colab={"base_uri": "https://localhost:8080/"} id="CuE6aIVaW-dE" outputId="754536bc-2dae-417f-9334-61cc921f8b5d"
# TRAINING SET
train_data['u_unique'] = train_data['User-ID'].map(train_data_user2idx)
train_data['i_unique'] = train_data['ISBN'].map(train_data_book2idx)

# TESTING SET
test_data['u_unique'] = test_data['User-ID'].map(test_data_user2idx)
test_data['i_unique'] = test_data['ISBN'].map(test_data_book2idx)

# Convert back to 3-column df
train_data = train_data[['u_unique', 'i_unique', 'Book-Rating']]
test_data = test_data[['u_unique', 'i_unique', 'Book-Rating']]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="yaiX-2kZW-S_" outputId="df0ee825-7875-4330-8cff-85c61268d135"
train_data.sample(5)
```

```python id="Cw5r1KxuW-M2"
n_users = train_data['u_unique'].nunique()
n_books = train_data['i_unique'].nunique()

train_matrix = np.zeros((n_users, n_books))

for entry in train_data.itertuples():                  # entry[1] is the user-id, entry[2] is the book-isbn
    train_matrix[entry[1]-1, entry[2]-1] = entry[3]    # -1 is to counter 0-based indexing
```

```python colab={"base_uri": "https://localhost:8080/"} id="Osk9RFL0W97w" outputId="eba51da3-ae79-4bc9-d998-a0bd405a8438"
train_matrix.shape
```

```python id="DFKU5ZooW939"
n_users = test_data['u_unique'].nunique()
n_books = test_data['i_unique'].nunique()

test_matrix = np.zeros((n_users, n_books))

for entry in test_data.itertuples():
    test_matrix[entry[1]-1, entry[2]-1] = entry[3]
```

```python colab={"base_uri": "https://localhost:8080/"} id="85QNEBr5W9zk" outputId="8049c913-72a5-4069-dba8-0a52455fecf1"
test_matrix.shape
```

```python id="xD5Dek5yW9u2"
train_matrix_small = train_matrix[:5000, :5000]
test_matrix_small = test_matrix[:5000, :5000]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_matrix_small, metric='cosine')
item_similarity = pairwise_distances(train_matrix_small.T, metric='cosine')
```

```python id="nxXgVY5wX2Fo"
def predict_books(ratings, similarity, type='user'): # default type is 'user'
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        
        # Use np.newaxis so that mean_user_rating has the same format as ratings
        
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
```

```python id="zY5yzby4X2B5"
item_prediction = predict_books(train_matrix_small, item_similarity, type='item')
user_prediction = predict_books(train_matrix_small, user_similarity, type='user')
```

```python colab={"base_uri": "https://localhost:8080/"} id="hsy_VTU1X192" outputId="4ccb092a-3c2d-4187-e972-4fa0d510b9fe"
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, test_matrix):
    prediction = prediction[test_matrix.nonzero()].flatten()
    test_matrix = test_matrix[test_matrix.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, test_matrix))

print(f'Item-based CF RMSE: {rmse(item_prediction, test_matrix_small)}')
print(f'User-based CF RMSE: {rmse(user_prediction, test_matrix_small)}')
```

```python colab={"base_uri": "https://localhost:8080/"} id="fjwAvzMPauGu" outputId="56e22ed3-fe94-4318-dbfd-5ea9d054ebd0"
pip install surprise
```

```python id="y65uu0zLX15F"
from surprise import Reader, Dataset

# Creating a 'Reader' object to set the limit of the ratings 

reader = Reader(rating_scale=(1, 10))

data = Dataset.load_from_df(ratings_explicit, reader)
```

```python colab={"base_uri": "https://localhost:8080/"} id="sDq-yLpTX11E" outputId="029f3d35-0c0c-4584-aa0f-023dddc0735a"
from surprise import SVD, model_selection, accuracy

model = SVD()

# Train on books dataset

%time model_selection.cross_validate(model, data, measures=['RMSE'], cv=5, verbose=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="o4sRh5m4X1xG" outputId="02bf1ba6-9063-479e-f4c1-326cae389f28"
trainset, testset = model_selection.train_test_split(data, test_size=0.2)

model = SVD()

model.fit(trainset)
predictions = model.test(testset)

accuracy.rmse(predictions)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EST1g14VW9qx" outputId="5883ae30-0617-4b8d-f5ba-ae806d51e99d"
uid = 276744  
iid = '038550120X' 
pred = model.predict(uid, iid, verbose=True)
```

```python id="LErSvCiWh2TH"
 #ratings_explicit[(ratings_explicit['User-ID']==pred.uid) & (ratings_explicit.ISBN==pred.iid)]['Book-Rating']
```

```python id="wX1TjcaciA8B"

```

```python colab={"base_uri": "https://localhost:8080/"} id="QXnvBoKnW9cU" outputId="ecae423c-d506-496e-9ce4-f1c0b9c98663"
print(f'The estimated rating for the book with ISBN code {pred.iid} from user #{pred.uid} is {pred.est:.2f}.\n')
#actual_rtg= ratings_explicit[(ratings_explicit['User-ID']==pred.uid) & 
#                             (ratings_explicit.ISBN==pred.iid)]['Book-Rating'].values[0]
#print(f'The real rating given for this was {actual_rtg:.2f}.')
```

```python id="gZdPHnjlYWMM"
# The following function was adapted from the surprise docs
# and can be used to get the top book recommendations for each user.
from collections import defaultdict

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
        
    return top_n
```

```python id="mHlDd3iJYfhe"
pred = model.test(testset)
top_n = get_top_n(pred)
```

```python id="QdSQ7wAwYfRb"
def get_reading_list(userid):
    """
    Retrieve full book titles from full 'books_users_ratings' dataframe
    """
    reading_list = defaultdict(list)
    top_n = get_top_n(pred, n=10)
    for n in top_n[userid]:
        book, rating = n
        title = new_book_df.loc[new_book_df.ISBN==book]['Book-Title'].unique()[0]
        reading_list[title] = rating
    return reading_list
```

```python colab={"base_uri": "https://localhost:8080/"} id="SiBFFy0gYdSw" outputId="4c9cd0e7-6529-48ce-eea5-7ab84907b132"
# Just take a random look at user_id=60337    116866
example_reading_list = get_reading_list(userid=60337)
for book, rating in example_reading_list.items():
    print(f'{book}: {rating}')
```

```python id="vuqJ8Su1YoCo"

```
