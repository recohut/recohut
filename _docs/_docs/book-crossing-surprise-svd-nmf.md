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
    language: python
    name: python3
---

<!-- #region id="g81xSGHXoGTa" -->
# Book-Crossing Recommendation System
> Book recommender system on book crossing dataset using surprise SVD and NMF models

- toc: true
- badges: true
- comments: true
- categories: [Surprise, SVD, NMF, Book]
- author: "<a href='https://github.com/tttgm/fellowshipai'>Tom McKenzie</a>"
- image:
<!-- #endregion -->

<!-- #region id="0WRq5pdzolD9" -->
## Setup
<!-- #endregion -->

```python id="k6TT0oKgoTuq" colab={"base_uri": "https://localhost:8080/"} outputId="0edb6221-33ab-4e1e-bb6b-51de0d2de477"
!pip install -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="FpuIkCPSoGTh"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import model_selection

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-white')
plt.rcParams.update({'font.size': 15})
%matplotlib inline
```

```python id="mScOsbySoxVt"
from recochef.datasets.bookcrossing import BookCrossing
```

<!-- #region id="T1RWAKSRoGTj" -->
## Load the dataset
<!-- #endregion -->

```python id="jg7VdI4roGTk"
bookcrossing = BookCrossing()
users = bookcrossing.load_users()
books = bookcrossing.load_items()
book_ratings = bookcrossing.load_interactions()
```

```python id="XD1h57n1oGTp" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="f0fa99a8-4da7-4e17-cec1-f86f5d641777"
users.head()
```

```python id="Vp5bMUj7oGTr" colab={"base_uri": "https://localhost:8080/", "height": 411} outputId="4ae25429-d6fd-4f43-8535-c7be3aebab92"
books.head()
```

```python id="eWGH3RIIoGTs" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="1b5ea488-9520-4cf7-ca71-dd38814aa499"
book_ratings.head()
```

```python id="hxXW-mEeoGTt" colab={"base_uri": "https://localhost:8080/"} outputId="460bb4b4-5315-40d1-9ffa-cade64753ae6"
print(f'Users: {len(users)}\nBooks: {len(books)}\nRatings: {len(book_ratings)}')
```

<!-- #region id="aHS8eNeqoGTu" -->
## EDA and Data cleaning
<!-- #endregion -->

<!-- #region id="3VR_65JZrTev" -->
### Users
<!-- #endregion -->

```python id="PsLTv-ctoGTx" colab={"base_uri": "https://localhost:8080/", "height": 142} outputId="8b964978-cfbf-45d6-faac-45638fcb9223"
users.describe(include='all').T
```

<!-- #region id="sC2xBAGIoGTx" -->
The age range goes from 0 to 244 years old! Obviously this cannot be correct; I'll set all ages less than 5 and older than 100 to NaN to try keep them realistic.
<!-- #endregion -->

```python id="F63Xy3XYoGTy"
users.loc[(users.AGE<5) | (users.AGE>100), 'AGE'] = np.nan
```

```python id="IU6NXlXMoGT1" colab={"base_uri": "https://localhost:8080/", "height": 489} outputId="76adc186-dfe0-4ffc-9dbd-53089e1e170e"
u = users.AGE.value_counts().sort_index()
plt.figure(figsize=(20, 10))
plt.bar(u.index, u.values)
plt.xlabel('Age')
plt.ylabel('counts')
plt.show()
```

<!-- #region id="dRSP0HFhoGT2" -->
Next, can we expand the 'Location' field to break it up into 'City', 'State', and 'Country'.
<!-- #endregion -->

<!-- #region id="OJNwLCGrq1cS" -->
> Note: Used Pandas Series.str.split method as it has an 'expand' parameter which can handle None cases
<!-- #endregion -->

```python id="LgVTuXJDoGT2" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="4653dfcf-26d0-4df9-efb7-3b1200d521e7"
user_location_expanded = users.LOCATION.str.split(',', 2, expand=True)
user_location_expanded.columns = ['CITY', 'STATE', 'COUNTRY']
users = users.join(user_location_expanded)
users.COUNTRY.replace('', np.nan, inplace=True)
users.drop(columns=['LOCATION'], inplace=True)
users.head()
```

<!-- #region id="5jHPKv6IoGT8" -->
### Books
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 199} id="bZOuv4z_rdW7" outputId="16f9781c-36b7-4c8e-e505-dba33aefcfa7"
books.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="A-AnXMmUr6Pl" outputId="f26eeac8-8fa6-4121-f8fd-c2b446784d39"
books.describe(include='all').T
```

```python id="QquU1U9_oGT_"
# Convert years to float
books.YEAR = pd.to_numeric(books.YEAR, errors='coerce')
```

```python id="-tqAjrujoGUA"
# Replace all years of zero with NaN
books.YEAR.replace(0, np.nan, inplace=True)
```

```python id="By-N2TRKoGUA" colab={"base_uri": "https://localhost:8080/", "height": 485} outputId="c9162c5d-c12b-47e6-e8b5-61d51aaf8c1d"
yr = books.YEAR.value_counts().sort_index()
yr = yr.where(yr>5) # filter out counts less than 5
plt.figure(figsize=(20, 10))
plt.bar(yr.index, yr.values)
plt.xlabel('Year of Publication')
plt.ylabel('counts')
plt.show()
```

<!-- #region id="yWQeBREhoGUA" -->
Note that in the plot above we filtered out counts less than 5, as there are a few books in the dataset with publication years in the 1300s, and a few in the future (?!). The plot above show the general trend that more recent books are much more frequent.

Let's take a look at some of those 'outlier' books. Maybe we'll even keep them as a separate dataset so we can filter them out if we need to later in the analysis. We'll leave them in for now, and then figure out how to handle them once we have more info later on.
<!-- #endregion -->

```python id="StMHoQEKoGUA" colab={"base_uri": "https://localhost:8080/"} outputId="3762babe-dce3-42fb-9b12-29c2802e33bf"
historical_books = books[books.YEAR<1900] # create df of old books
books_from_the_future = books[books.YEAR>2018] # create df of books with publication yrs in the future!

hist_books_mini = historical_books[['TITLE', 'YEAR']]
future_books_mini = books_from_the_future[['TITLE', 'YEAR']]
print(f'Historical books:\n{hist_books_mini}')
print('\n')
print(f'Future books:\n{future_books_mini}')
```

<!-- #region id="bLPouO9LoGUB" -->
I think we can probably omit the 'historical_books' as they may potentially skew the model and do not seem to have much relevance to the wider userbase.

Some of the 'future' books actually appear to be errors (e.g. Alice in Wonderland, Edgar Allen Poe, etc.)... Perhaps they were supposed to be e.g. 1950 instead of 2050? However, instead of investigating this further, since there are <20 books here I will simply remove them from the 'books' table.
<!-- #endregion -->

```python id="GVkbdBDHoGUB" colab={"base_uri": "https://localhost:8080/"} outputId="eb824cb7-676d-4d57-acaa-4f3b36a7b44a"
print(f'Length of books dataset before removal: {len(books)}')
books = books.loc[~(books.ITEMID.isin(historical_books.ITEMID))] # remove historical books
books = books.loc[~(books.ITEMID.isin(books_from_the_future.ITEMID))] # remove historical books
print(f'Length of books dataset after removal: {len(books)}')
```

<!-- #region id="sc8TfMm6oGUB" -->
We clean up the ampersand formatting in the Publisher field.
<!-- #endregion -->

```python id="DRXOBXkVoGUC" colab={"base_uri": "https://localhost:8080/", "height": 411} outputId="ecc24122-61b9-45e8-dc5b-16f6f02b83fe"
books.PUBLISHER = books.PUBLISHER.str.replace('&amp', '&', regex=False)
books.head()
```

<!-- #region id="rJd3lqojoGUC" -->
Check that there are no duplicated book entries.
<!-- #endregion -->

```python id="8cxxbmLDoGUD" colab={"base_uri": "https://localhost:8080/"} outputId="558c1540-70c6-46bf-cbed-a3c2507c1aa4"
uniq_books = books.ITEMID.nunique()
all_books = books.ITEMID.count()
print(f'No. of unique books: {uniq_books} | All book entries: {all_books}')
```

<!-- #region id="yIbT7YzooGUE" -->
Let's look at the most frequent Publishing houses in the dataset.
<!-- #endregion -->

```python id="qtz7MiXnoGUE" colab={"base_uri": "https://localhost:8080/"} outputId="7c1b54d5-e212-4878-c52b-ebf49a5b3e7d"
top_publishers = books.PUBLISHER.value_counts()[:10]
print(f'The 10 publishers with the most entries in the books table are:\n{top_publishers}')
```

<!-- #region id="5nhrv4O3oGUF" -->
What about authors with the most entries?
<!-- #endregion -->

```python id="FTcGa8HEoGUG" colab={"base_uri": "https://localhost:8080/"} outputId="3f1b8c6b-9da5-4f14-e411-f4454448f806"
top_authors = books.AUTHOR.value_counts()[:10]
print(f'The 10 authors with the most entries in the books table are:\n{top_authors}')
```

<!-- #region id="yPNmWRkqoGUH" -->
We should search for empty or NaN values in these fields too.
<!-- #endregion -->

```python id="TmbpfUPSoGUH" colab={"base_uri": "https://localhost:8080/"} outputId="de320a50-b35d-49e4-f0da-61947587665a"
empty_string_publisher = books[books.PUBLISHER == ''].PUBLISHER.count()
nan_publisher = books.PUBLISHER.isnull().sum()
print(f'There are {empty_string_publisher} entries with empty strings, and {nan_publisher} NaN entries in the Publisher field')
```

<!-- #region id="4kkz8MJfoGUI" -->
Great - no empty strings in the Publisher field, and only 2 NaNs.
<!-- #endregion -->

```python id="_FzRibRfoGUJ" colab={"base_uri": "https://localhost:8080/"} outputId="3d91db0a-1381-4ad5-e50d-988506386010"
empty_string_author = books[books.AUTHOR == ''].AUTHOR.count()
nan_author = books.AUTHOR.isnull().sum()
print(f'There are {empty_string_author} entries with empty strings, and {nan_author} NaN entries in the Author field')
```

<!-- #region id="h74pnJO2oGUK" -->
Cool, only 1 NaN in the Author field.

Let's look at the titles.
<!-- #endregion -->

```python id="ZKUi0je5oGUK" colab={"base_uri": "https://localhost:8080/"} outputId="968d5fdd-2dec-4659-a86e-1788c2fc4149"
top_titles = books.TITLE.value_counts()[:10]
print(f'The 10 book titles with the most entries in the books table are:\n{top_titles}')
```

<!-- #region id="gE1JabzPoGUL" -->
This is actually quite an important observation. Although all of the ISBN entries are *unique* in the 'books' dataframe, different *forms* of the **same** book will have different ISBNs - i.e. paperback, e-book, etc. Therefore, we can see that some books have multiple ISBN entries (e.g. Jane Eyre has 19 different ISBNs, each corresponding to a different version of the book).

Let's take a look at, for example, the entries for 'Jane Eyre'.
<!-- #endregion -->

```python id="cX7HjwSgoGUM" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="6a92810b-e9a7-4703-edd9-f5a4bf297416"
books[books.TITLE=='Jane Eyre']
```

<!-- #region id="44C2ZhdxoGUN" -->
It looks like each ISBN assigned to the book 'Jane Eyre' has different Publisher and Year of Publication values also.

It might be more useful for our model if we simplified this to give each book a *unique* identifier, independent of the book format, as our recommendations will be for a book, not a specific version of a book. Therefore, all values in the Jane Eyre example above would stay the same, except all of the Jane Eyre entries would additionally be assigned a *unique ISBN* code as a new field.

**Will create this more unique identifier under the field name 'UNIQUE_ITEMIDS'. Note that entries with only a single ISBN number will be left the same. However, will need to do this after joining to the other tables in the dataset, as some ISBNs in the 'book-rating' table may be removed if done prior.**
<!-- #endregion -->

<!-- #region id="e5lQCkVtoGUN" -->
### Interactions
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="zR9FKnFltkQp" outputId="021e7429-c2cb-48fe-8b80-ecec2facd7cc"
book_ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 142} id="q_q5G0yttpJ0" outputId="0fa4e956-ab70-4dcd-c2d4-fe213303c784"
book_ratings.describe(include='all').T
```

```python colab={"base_uri": "https://localhost:8080/"} id="eUaA5hVft3Yt" outputId="bcda2197-b7ad-4915-f103-d668bf2daab4"
book_ratings.dtypes
```

<!-- #region id="ShKPkSxvoGUQ" -->
The data types already look good. Remember that the ISBN numbers may contain letters, and so should be left as strings.

Which users contribute the most ratings?
<!-- #endregion -->

```python id="Iznlk3C9oGUQ" colab={"base_uri": "https://localhost:8080/"} outputId="28e52faa-a0f1-47a3-cc9f-52d980ddf9a6"
super_users = book_ratings.groupby('USERID').ITEMID.count().sort_values(ascending=False)
print(f'The 20 users with the most ratings:\n{super_users[:20]}')
```

<!-- #region id="9IiQu8wjoGUR" -->
Wow! User \#11676 has almost twice as many ratings as the next highest user! All of the top 20 users have thousands of ratings, which seems like a lot, although maybe I'm just a slow reader...

Let's see how they are distributed.
<!-- #endregion -->

```python id="-_eu_HvwoGUR" colab={"base_uri": "https://localhost:8080/", "height": 284} outputId="a30f2e63-2f40-4e63-c421-633743bf48d4"
# user distribution - users with more than 50 ratings removed
user_hist = super_users.where(super_users<50)
user_hist.hist(bins=30)
plt.xlabel('No. of ratings')
plt.ylabel('count')
plt.show()
```

<!-- #region id="PbNYApFVoGUS" -->
It looks like **_by far_** the most frequent events are users with only 1 or 2 rating entries. We can see that the 'super users' with thousands of ratings are significant outliers.

This becomes clear if we make the same histogram with a cutoff for users with a minimum of 1000 ratings.
<!-- #endregion -->

```python id="5b5H0JwtoGUS" colab={"base_uri": "https://localhost:8080/", "height": 284} outputId="6b7822cb-b420-4eb9-c055-c10a7e349957"
# only users with more than 1000 ratings
super_user_hist = super_users.where(super_users>1000)
super_user_hist.hist(bins=30)
plt.xlabel('No. of ratings (min. 1000)')
plt.ylabel('count')
plt.show()
```

<!-- #region id="2MSB1-u_oGUT" -->
Let's see what the distribution of **ratings** looks like.
<!-- #endregion -->

```python id="5XxhxDRRoGUT" colab={"base_uri": "https://localhost:8080/", "height": 338} outputId="66a3148d-bcb0-48d2-8384-b9ab33fe2f99"
rtg = book_ratings.RATING.value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(rtg.index, rtg.values)
plt.xlabel('Rating')
plt.ylabel('counts')
plt.show()
```

<!-- #region id="rLmBa7G-oGUT" -->
Seems like most of the entries have a rating of zero!

After doing some research on the internet regarding this (and similar) datasets, it appears that the rating scale is actually from 1 to 10, and a 0 indicates an 'implicit' rather than an 'explicit' rating. An implicit rating represents an interaction (may be positive or negative) between the user and the item. Implicit interactions usually need to be handled differently from explicit ones.

For the modeling step we'll only be looking at *explicit* ratings, and so the 0 rating entry rows will be removed.
<!-- #endregion -->

```python id="7nMrDIZeoGUU" colab={"base_uri": "https://localhost:8080/"} outputId="feca9ce4-5ad6-4b7c-a346-bfd08a473681"
print(f'Size of book_ratings before removing zero ratings: {len(book_ratings)}')
book_ratings = book_ratings[book_ratings.RATING != 0]
print(f'Size of book_ratings after removing zero ratings: {len(book_ratings)}')
```

<!-- #region id="7mROHsCooGUV" -->
By removing the implicit ratings we have reduced our sample size by more than half.

Let's look at how the ratings are distributed again.
<!-- #endregion -->

```python id="JCezmwp1oGUV" colab={"base_uri": "https://localhost:8080/", "height": 338} outputId="189d23c0-aed2-487b-8b58-8e10c9a621ad"
rtg = book_ratings.RATING.value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(rtg.index, rtg.values)
plt.xlabel('Rating')
plt.ylabel('counts')
plt.show()
```

<!-- #region id="39a_qmEooGUW" -->
This is much more clear! Now we can see that 8 is the most frequent rating, while users tend to give ratings > 5, with very few low ratings given.
<!-- #endregion -->

<!-- #region id="k-bwINpNoGUW" -->
### Merge
<!-- #endregion -->

<!-- #region id="vME5EfygoGUX" -->
First, we'll join the 'books' table to the 'book_ratings' table on the ISBN field.
<!-- #endregion -->

```python id="l37nt9xCoGUX" colab={"base_uri": "https://localhost:8080/"} outputId="f67dc875-a9a6-4b92-d902-cecebf929666"
print(f'Books table size: {len(books)}')
print(f'Ratings table size: {len(book_ratings)}')
books_with_ratings = book_ratings.join(books.set_index('ITEMID'), on='ITEMID')
print(f'New table size: {len(books_with_ratings)}')
```

<!-- #region id="X9wjDlHPoGUX" -->
Let's take a look at the new table.
<!-- #endregion -->

```python id="VDSdI_iRoGUY" colab={"base_uri": "https://localhost:8080/", "height": 360} outputId="da99d860-230d-4693-a4c7-95896118b46a"
books_with_ratings.head()
```

```python id="hRtbcS7BoGUY" colab={"base_uri": "https://localhost:8080/"} outputId="05ac86f9-11b9-4c2c-9b75-4d0237b3ad54"
print(f'There are {books_with_ratings.TITLE.isnull().sum()} books with no title/author information.')
print(f'This represents {len(books_with_ratings)/books_with_ratings.TITLE.isnull().sum():.2f}% of the ratings dataset.')
```

<!-- #region id="ylYa8nsOoGUZ" -->
There seems to be quite a few ISBNs in the ratings table that did not match an ISBN in the books table, almost 9% of all entries!

There isn't really anything we can do about that, but we should really remove them from the dataset as we won't be able to access the title of the book to make a recommendation even if the model can use them.
<!-- #endregion -->

```python id="Kb6oMsIpoGUZ" colab={"base_uri": "https://localhost:8080/"} outputId="2b7a6291-bf92-4307-985b-22eaa43a1849"
books_with_ratings.info()
```

<!-- #region id="bSrMjH88oGUZ" -->
It looks like the ```year_of_publication``` field contains the most NaN entries, while ```USERID```, ```isbn```, and ```book_rating``` are full. The ```book_title```, ```book_author```, and ```publisher``` fields contain approximately the same number of missing entries.

We'll choose to remove rows for which the ```book_title``` is empty, as this is the most crucial piece of data needed to identify the book.
<!-- #endregion -->

```python id="Z5YQjFK_oGUZ"
books_with_ratings.dropna(subset=['TITLE'], inplace=True) # remove rows with missing title/author data
```

<!-- #region id="kGVocZyNoGUa" -->
Let's see which books have the highest **cumulative** book rating values.
<!-- #endregion -->

```python id="3DBSqSGtoGUa" colab={"base_uri": "https://localhost:8080/", "height": 740} outputId="306a4122-b48d-45c1-ecc6-54dae2fe840e"
cm_rtg = books_with_ratings.groupby('TITLE').RATING.sum()
cm_rtg = cm_rtg.sort_values(ascending=False)[:10]
idx = cm_rtg.index.tolist() # Get sorted book titles
vals = cm_rtg.values.tolist() # Get corresponding cm_rtg values

plt.figure(figsize=(10, 5))
plt.bar(range(len(idx)), vals)
plt.xticks(range(len(idx)), idx, rotation='vertical')
plt.ylabel('cumulative rating score')
plt.show()
```

<!-- #region id="4waQMETRoGUa" -->
This seems about right as it combines the total number of ratings with the score given, so these are all really popular book titles.

What about the highest **average ratings** (with a minimum of at least 50 ratings recieved)?
<!-- #endregion -->

```python id="1RQTrD7uoGUa" colab={"base_uri": "https://localhost:8080/"} outputId="da415e71-c724-4ab0-b35a-dac46936354b"
cutoff = books_with_ratings.TITLE.value_counts()
mean_rtg = books_with_ratings[books_with_ratings.TITLE.isin(cutoff[cutoff>50].index)].groupby('TITLE')['RATING'].mean()
mean_rtg.sort_values(ascending=False)[:10] # show only top 10
```

<!-- #region id="_UhdB4LMoGUb" -->
This looks perfectly reasonable. The Harry Potter and Lord of the Rings books rate extremely highly, as expected.

How about the **lowest-rated** books?
<!-- #endregion -->

```python id="XypDVpvQoGUb" colab={"base_uri": "https://localhost:8080/"} outputId="87f62597-1637-4643-c9b8-e5af135387b9"
mean_rtg.sort_values(ascending=False)[-10:] # bottom 10 only
```

<!-- #region id="CGHcAfi_oGUc" -->
Seems like the *lowest average* rating in the dataset is only a 4.39 - and all the rest of the books have average ratings higher than 5.

I haven't heard of any of these books, so I can't really comment on if they seem correct here.

**Now I'd like to tackle the challenge of the same book potentially having multiple ISBN numbers (for the different formats it is available in). We should clean that up here before we add the 'user' table.**
<!-- #endregion -->

<!-- #region id="ObMGSilzoGUc" -->
### Single ISBN per book
Restrict books to a "single ISBN per book" (regardless of format)
<!-- #endregion -->

<!-- #region id="66Q1S9S2oGUd" -->
Let's look again at the book titles which have the most associated ISBN numbers.
<!-- #endregion -->

```python id="VwSgGH5-oGUe" colab={"base_uri": "https://localhost:8080/"} outputId="cf37e7c9-2996-45bb-89a3-b7254e546a39"
books_with_ratings.groupby('TITLE').ITEMID.nunique().sort_values(ascending=False)[:10]
```

```python id="z4-pQebwoGUe" colab={"base_uri": "https://localhost:8080/"} outputId="25f3b765-4b8e-491c-fca3-2ab92f0b36ba"
multiple_isbns = books_with_ratings.groupby('TITLE').ITEMID.nunique()
multiple_isbns.value_counts()
```

<!-- #region id="6tRLSnaioGUf" -->
We can see that the vast majority of books have less only 1 associated ISBN number, however quite a few multiple ISBNs. We want to create a ```UNIQUE_ITEMIDS``` such that a single book will only have 1 identifier when fed to the recommendation model.
<!-- #endregion -->

```python id="4tg1y16loGUf"
has_mult_isbns = multiple_isbns.where(multiple_isbns>1)
has_mult_isbns.dropna(inplace=True) # remove NaNs, which in this case is books with a single ISBN number
```

```python id="IEPz3bGkoGUg" colab={"base_uri": "https://localhost:8080/"} outputId="77a227e4-1cf9-4b24-b39c-22a0b419dffe"
print(f'There are {len(has_mult_isbns)} book titles with multiple ISBN numbers which we will try to re-assign to a unique identifier')
```

```python id="pf1kBD0toGUh" colab={"base_uri": "https://localhost:8080/"} outputId="9b27f6b6-e0e2-4ef8-b021-35d21362bb87"
# Check to see that our friend Jane Eyre still has multiple ISBN values
has_mult_isbns['Jane Eyre']
```

<!-- #region id="HCUw-HbboGUh" -->
**Note:** Created the dictionary below and pickled it, just need to load it again (or run it if the first time on a new system).
<!-- #endregion -->

```python id="v4zvRDsmoGUi" colab={"base_uri": "https://localhost:8080/"} outputId="ba28c30a-4026-4931-e8f9-259e6c21936d"
# Create dictionary for books with multiple isbns
def make_isbn_dict(df):
    title_isbn_dict = {}
    for title in has_mult_isbns.index:
        isbn_series = df.loc[df.TITLE==title].ITEMID.unique() # returns only the unique ISBNs
        title_isbn_dict[title] = isbn_series.tolist()
    return title_isbn_dict

%time dict_UNIQUE_ITEMIDS = make_isbn_dict(books_with_ratings)

# As the loop takes a while to run (8 min on the full dataset), pickle this dict for future use
with open('multiple_isbn_dict.pickle', 'wb') as handle:
    pickle.dump(dict_UNIQUE_ITEMIDS, handle, protocol=pickle.HIGHEST_PROTOCOL)

# LOAD isbn_dict back into namespace
with open('multiple_isbn_dict.pickle', 'rb') as handle:
    multiple_isbn_dict = pickle.load(handle)
```

```python id="JpnZDtMWoGUk" colab={"base_uri": "https://localhost:8080/"} outputId="fca551fb-5e1b-4c3d-933e-72d2399d3161"
print(f'There are now {len(multiple_isbn_dict)} books in the ISBN dictionary that have multiple ISBN numbers')
```

<!-- #region id="BIc4o0l_oGUk" -->
Let's take a quick look in the dict we just created for the 'Jane Eyre' entry - it should contain a list of 14 ISBN numbers.
<!-- #endregion -->

```python id="nmxlxzFOoGUl" colab={"base_uri": "https://localhost:8080/"} outputId="683f0072-677f-4a3b-efcb-8cb1561b2f6f"
print(f'Length of Jane Eyre dict entry: {len(multiple_isbn_dict["Jane Eyre"])}\n')
multiple_isbn_dict['Jane Eyre']
```

<!-- #region id="EI3KXjIvoGUm" -->
Looking good!

As I don't really know what each of the different ISBN numbers refers to (from what I understand the code actually signifies various things including publisher, year, type of print, etc, but decoding this is outside the scope of this analysis), I'll just select the **first** ISBN number that appears in the list of values to set as our ```UNIQUE_ITEMIDS``` for that particular book.

_**Note**_: ISBN numbers are currently 13 digits long, but used to be 10. Any ISBN that isn't 10 or 13 digits long is probably an error that should be handled somehow. Any that are 9 digits long might actually be SBN numbers (pre-1970), and can be converted into ISBN's by just pre-fixing with a zero.
<!-- #endregion -->

```python id="vHPFNFz-oGUm" colab={"base_uri": "https://localhost:8080/"} outputId="ceee7f14-fee4-4e50-e8a7-1b6498ebc69f"
# Add 'UNIQUE_ITEMIDS' column to 'books_with_ratings' dataframe that includes the first ISBN if multiple ISBNS,
# or just the ISBN if only 1 ISBN present anyway.
def add_UNIQUE_ITEMIDS_col(df):
    df['UNIQUE_ITEMIDS'] = df.apply(lambda row: multiple_isbn_dict[row.TITLE][0] if row.TITLE in multiple_isbn_dict.keys() else row.ITEMID, axis=1)
    return df

%time books_with_ratings = add_UNIQUE_ITEMIDS_col(books_with_ratings)
```

```python id="AJR3omsYoGUm" colab={"base_uri": "https://localhost:8080/", "height": 394} outputId="b8f29706-3f07-45f8-c042-380f1b82ad23"
books_with_ratings.head()
```

<!-- #region id="S7ifEoYGoGUn" -->
The table now includes our ```UNIQUE_ITEMIDS``` field.

Let's check to see if the 'Jane Eyre' entries have been assigned the ISBN '1590071212', which was the first val in the dictionary for this title.
<!-- #endregion -->

```python id="PNyn8kALoGUn" colab={"base_uri": "https://localhost:8080/", "height": 360} outputId="b51d68a2-b16a-4c27-8590-cb7ce02ba2e7"
books_with_ratings[books_with_ratings.TITLE=='Jane Eyre'].head()
```

<!-- #region id="Vs5T6u5SoGUn" -->
Great! Seems to have worked well.

We won't replace the original ISBN column with the 'UNIQUE_ITEMIDS' column, but just note that the recommendation model should be based on the 'UNIQUE_ITEMIDS' field.
<!-- #endregion -->

<!-- #region id="dWACLR_-zwcz" -->
### Remove Small and Large book-cover URL columns
<!-- #endregion -->

```python id="IieQMNRKz5u-"
books_users_ratings.drop(['URLSMALL', 'URLLARGE'], axis=1, inplace=True)
```

<!-- #region id="epRywx7goGUn" -->
## Join the 'users' table on the 'USERID' field
<!-- #endregion -->

```python id="s8tAP81ioGUn" colab={"base_uri": "https://localhost:8080/"} outputId="2f04807b-cb3e-4eed-d26a-936333e8a198"
print(f'Books+Ratings table size: {len(books_with_ratings)}')
print(f'Users table size: {len(users)}')
books_users_ratings = books_with_ratings.join(users.set_index('USERID'), on='USERID')
print(f'New "books_users_ratings" table size: {len(books_users_ratings)}')
```

<!-- #region id="Sh9rWgSvoGUo" -->
Inspect the new table.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="KVNXMrgzzqBm" outputId="1f76a837-ab7e-492e-a9d8-cde10d974d1a"
books_users_ratings.head()
```

```python id="dfRTU2x1oGUq" colab={"base_uri": "https://localhost:8080/"} outputId="2e5dcce8-21e6-40fb-9426-b2cc408236d8"
books_users_ratings.info()
```

<!-- #region id="jGgBFquqoGUr" -->
There are a few missing ```age```, ```year_of_publication```, ```publisher```, and ```country``` entries, but the primary fields of ```USERID```, ```UNIQUE_ITEMIDS```, and ```book_rating``` are all full, which is good.

In terms of the data types, ```USERID``` and ```book_rating``` are integers, while the ```UNIQUE_ITEMIDS``` are strings (which is expected as the ISBN numbers may also contain letters).
<!-- #endregion -->

```python id="ERFxFR5WoGUr" colab={"base_uri": "https://localhost:8080/"} outputId="2ef86aff-c9ea-4e39-a815-a660f8a0aac6"
books_users_ratings.shape
```

<!-- #region id="_nwRSsDioGUr" -->
## Recommender model

Collaborative filtering use similarities of the 'user' and 'item' fields, with values of 'rating' predicted based on either user-item, or item-item similarity:
 - Item-Item CF: "Users who liked this item also liked..."
 - User-Item CF: "Users who are similar to you also liked..."
 
In both cases, we need to create a user-item matrix built from the entire dataset. We'll create a matrix for each of the training and testing sets, with the users as the rows, the books as the columns, and the rating as the matrix value. Note that this will be a very sparse matrix, as not every user will have watched every movie etc.

We'll first create a new dataframe that contains only the relevant columns (```USERID```, ```UNIQUE_ITEMIDS```, and ```book_rating```).
<!-- #endregion -->

```python id="YmrbxBXkoGUs" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="a3137809-4e9d-4988-8325-da5f8b973705"
user_item_rating = books_users_ratings[['USERID', 'UNIQUE_ITEMIDS', 'RATING']]
user_item_rating.head()
```

<!-- #region id="mCOER6kBoGUs" -->
We know what the distribution of ratings should look like (as we plotted it earlier) - let's plot it again on this new dataframe to just quickly check that it looks right.
<!-- #endregion -->

```python id="f_nFLtvyoGUs" colab={"base_uri": "https://localhost:8080/", "height": 338} outputId="3e73302f-d61a-4236-8859-677c049f6dab"
rtg = user_item_rating.RATING.value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(rtg.index, rtg.values)
plt.xlabel('Rating')
plt.ylabel('counts')
plt.show()
```

<!-- #region id="ndlUjRBuoGUs" -->
Looks perfect! Continue.
<!-- #endregion -->

<!-- #region id="WgPhxeeEoGUt" -->
### Using ```sklearn``` to generate training and testing subsets
<!-- #endregion -->

```python id="BqFHawMNoGUt"
train_data, test_data = model_selection.train_test_split(user_item_rating, test_size=0.20)
```

```python id="45X1w_ZpoGUt" colab={"base_uri": "https://localhost:8080/"} outputId="37a20cc6-d6d1-4581-bec6-ed8596bd7484"
print(f'Training set size: {len(train_data)}')
print(f'Testing set size: {len(test_data)}')
print(f'Test set is {(len(test_data)/(len(train_data)+len(test_data))*100):.0f}% of the full dataset.')
```

<!-- #region id="1mKeEe9toGUu" -->
### Map the ```USERID``` and ```UNIQUE_ITEMIDS``` fields to sequential integers for matrix processing
<!-- #endregion -->

```python id="2uBBs55PoGUu"
### TRAINING SET
# Get int mapping for USERID
u_unique_train = train_data.USERID.unique()  # create a 'set' (i.e. all unique) list of vals
train_data_user2idx = {o:i for i, o in enumerate(u_unique_train)}
# Get int mapping for UNIQUE_ITEMIDS
b_unique_train = train_data.UNIQUE_ITEMIDS.unique()  # create a 'set' (i.e. all unique) list of vals
train_data_book2idx = {o:i for i, o in enumerate(b_unique_train)}

### TESTING SET
# Get int mapping for USERID
u_unique_test = test_data.USERID.unique()  # create a 'set' (i.e. all unique) list of vals
test_data_user2idx = {o:i for i, o in enumerate(u_unique_test)}
# Get int mapping for UNIQUE_ITEMIDS
b_unique_test = test_data.UNIQUE_ITEMIDS.unique()  # create a 'set' (i.e. all unique) list of vals
test_data_book2idx = {o:i for i, o in enumerate(b_unique_test)}
```

```python id="KTV6nobIoGUv"
### TRAINING SET
train_data['USER_UNIQUE'] = train_data['USERID'].map(train_data_user2idx)
train_data['ITEM_UNIQUE'] = train_data['UNIQUE_ITEMIDS'].map(train_data_book2idx)

### TESTING SET
test_data['USER_UNIQUE'] = test_data['USERID'].map(test_data_user2idx)
test_data['ITEM_UNIQUE'] = test_data['UNIQUE_ITEMIDS'].map(test_data_book2idx)

### Convert back to 3-column df
train_data = train_data[['USER_UNIQUE', 'ITEM_UNIQUE', 'RATING']]
test_data = test_data[['USER_UNIQUE', 'ITEM_UNIQUE', 'RATING']]
```

```python id="Rowne8d9oGUv" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="958183f0-dfda-4d98-f8b7-9e1e800e4516"
train_data.tail()
```

```python id="dKLH4N1roGUw" colab={"base_uri": "https://localhost:8080/"} outputId="069046c3-e346-4fac-c8d9-3f72209d1f46"
train_data.dtypes
```

<!-- #region id="QnrjZIy3oGUx" -->
This dataset is now ready to be processed via a collaborative filtering approach!

**Note:** When we need to identify the user or book from the model we'll need to refer back to the ```train_data_user2idx``` and ```train_data_book2idx``` dictionaries to locate the ```USERID``` and ```UNIQUE_ITEMIDS```, respectively.
<!-- #endregion -->

```python id="QoGNxJtmoGUy"
### TRAINING SET
# Create user-item matrices
n_users = train_data['USER_UNIQUE'].nunique()
n_books = train_data['ITEM_UNIQUE'].nunique()

# First, create an empty matrix of size USERS x BOOKS (this speeds up the later steps)
train_matrix = np.zeros((n_users, n_books))

# Then, add the appropriate vals to the matrix by extracting them from the df with itertuples
for entry in train_data.itertuples(): # entry[1] is the user-id, entry[2] is the book-isbn
    train_matrix[entry[1]-1, entry[2]-1] = entry[3] # -1 is to counter 0-based indexing
```

```python id="d8rveQiloGUy" outputId="242ac063-3b18-40e2-8158-91ba8beeba72"
train_matrix.shape
```

<!-- #region id="D02jgTG0oGUy" -->
Now do the same for the test set.
<!-- #endregion -->

```python id="92YKvmBqoGUz"
### TESTING SET
# Create user-item matrices
n_users = test_data['u_unique'].nunique()
n_books = test_data['b_unique'].nunique()

# First, create an empty matrix of size USERS x BOOKS (this speeds up the later steps)
test_matrix = np.zeros((n_users, n_books))

# Then, add the appropriate vals to the matrix by extracting them from the df with itertuples
for entry in test_data.itertuples(): # entry[1] is the user-id, entry[2] is the book-isbn
    test_matrix[entry[1]-1, entry[2]-1] = entry[3] # -1 is to counter 0-based indexing
```

```python id="1B0AGzKZoGUz" outputId="c1ab63df-a5a5-4948-ce92-d1818ecf03b5"
test_matrix.shape
```

<!-- #region id="k_jqjCLqoGUz" -->
Now the matrix is in the correct format, with the user and book entries encoded from the mapping dict created above!
<!-- #endregion -->

<!-- #region id="Hg2f-URCoGUz" -->
### Calculating cosine similarity with the 'pairwise distances' function

To determine the similarity between users/items we'll use the 'cosine similarity' which is a common n-dimensional distance metric.

**Note:** since all of the rating values are positive (1-10 scale), the cosine distances will all fall between 0 and 1.
<!-- #endregion -->

```python id="Y2p4DcX9oGU0"
# It may take a while to calculate, so I'll perform on a subset initially
train_matrix_small = train_matrix[:10000, :10000]
test_matrix_small = test_matrix[:10000, :10000]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_matrix_small, metric='cosine')
item_similarity = pairwise_distances(train_matrix_small.T, metric='cosine') # .T transposes the matrix (NumPy)
```

<!-- #region id="ChR4tsumoGU0" -->
If we are looking at similarity between users we need to account for the average behaviour of that individual user. For example, one user may give all movies quite high ratings, whereas one might give all ratings between 3 and 7. These users might otherwise have quite similar preferences.

To do this, we use the users average rating as a 'weighting' factor.

If we are looking at item-based similarity we don't need to add this weighting factor.

We can incorporate this into a ```predict()``` function, like so:
<!-- #endregion -->

```python id="qMwMzqGEoGU0"
def predict(ratings, similarity, type='user'): # default type is 'user'
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has the same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
```

<!-- #region id="c785znVioGU0" -->
Then can make our predictions!
<!-- #endregion -->

```python id="tAYN1_DhoGU0"
item_prediction = predict(train_matrix_small, item_similarity, type='item')
user_prediction = predict(train_matrix_small, user_similarity, type='user')
```

<!-- #region id="YRWMWY3MoGU1" -->
### Evaluation

How do we know if this is making good ```rating``` predictions?

We'll start by just taking the root mean squared error (RMSE) (from ```sklearn```) of predicted values in the ```test_set``` (i.e. where we know what the answer should be).

Since we want to compare only predicted ratings that are in the test set, we can filter out all other predictions that aren't in the test matrix.
<!-- #endregion -->

```python id="9D3w-2UfoGU1" outputId="b10a2ccb-d2d6-422c-a7e3-89fae268ab96"
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, test_matrix):
    prediction = prediction[test_matrix.nonzero()].flatten()
    test_matrix = test_matrix[test_matrix.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, test_matrix))

# Call on test set to get error from each approach ('user' or 'item')
print(f'User-based CF RMSE: {rmse(user_prediction, test_matrix_small)}')
print(f'Item-based CF RMSE: {rmse(item_prediction, test_matrix_small)}')
```

<!-- #region id="9OEPWAdloGU1" -->
For the user-item and the item-item recommendations we get RMSE = 7.85 (MSE > 60) for both. This is pretty bad, but we only trained over a small subset of the data.

Although this collaborative filtering setup is relatively simple to write, it doesn't scale very well at all, as it is all stored in memory! (Hence why we only used a subset of the training/testing data).

----------------

Instead, we should really use a model-based (based on matrix factorization) recommendation algorithm. These are inherently more scalable and can deal with higher sparsity level than memory-based models, and are considered more powerful due to their ability to pick up on "latent factors" in the relationships between what sets of items users like. However, they still suffer from the "cold start" problem (where a new user has no history).

Fortunately, there is a Python library called ```surprise``` that was built specifically for the implementation of model-based recommendation systems! This library comes with many of the leading algorithms in this space already built-in. Let's try use it for our book recommender system.
<!-- #endregion -->

<!-- #region id="oHPIpMiioGU2" -->
# Using the ```surprise``` library for building a recommender system
Several common model-based algorithms including SVD, KNN, and non-negative matrix factorization are built-in!  
See [here](http://surprise.readthedocs.io/en/stable/getting_started.html#basic-usage) for the docs.
<!-- #endregion -->

```python id="5fVOhGz_oGU2"
from surprise import Reader, Dataset
```

```python id="3EFO4YtxoGU2" outputId="1598146b-de61-4c49-e10a-9382da6511f7"
user_item_rating.head() # take a look at our data
```

```python id="Id9lx_-yoGU3"
# First need to create a 'Reader' object to set the scale/limit of the ratings field
reader = Reader(rating_scale=(1, 10))

# Load the data into a 'Dataset' object directly from the pandas df.
# Note: The fields must be in the order: user, item, rating
data = Dataset.load_from_df(user_item_rating, reader)
```

```python id="Xbt8Kj0qoGU3"
# Load the models and 'evaluation' method
from surprise import SVD, NMF, model_selection, accuracy
```

<!-- #region id="pnJkjUAFoGU3" -->
Where: SVD = Singular Value Decomposition (orthogonal factorization), NMF = Non-negative Matrix Factorization.

**Note** that when using the ```surprise``` library we don't need to manually create the mapping of USERID and UNIQUE_ITEMIDS to integers in a custom dict. See [here](http://surprise.readthedocs.io/en/stable/FAQ.html#raw-inner-note) for details. 
<!-- #endregion -->

<!-- #region id="L9F7XC8zoGU4" -->
### SVD model
<!-- #endregion -->

<!-- #region id="09a67_HjoGU4" -->
**_Using cross-validation (5 folds)_**
<!-- #endregion -->

```python id="hpvvdUQwoGU4" outputId="5eb7a621-9c8b-4514-ba23-ecf2352f4cbe"
# Load SVD algorithm
model = SVD()

# Train on books dataset
%time model_selection.cross_validate(model, data, measures=['RMSE'], cv=5, verbose=True)
```

<!-- #region id="RPkvHMEFoGU4" -->
The SVD model gives an average RMSE of ca. 1.64 after 5-folds, with a fit time of ca. 28 s for each fold.
<!-- #endregion -->

<!-- #region id="D40FxOqdoGU5" -->
**_Using test-train split_**
<!-- #endregion -->

```python id="TJW0wVY9oGU5" outputId="a168b637-922b-44f3-c1a8-a0a28ec3f2e2"
# set test set to 20%.
trainset, testset = model_selection.train_test_split(data, test_size=0.2)

# Instantiate the SVD model.
model = SVD()

# Train the algorithm on the training set, and predict ratings for the test set
model.fit(trainset)
predictions = model.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)
```

<!-- #region id="OJNdz1qwoGU5" -->
Using a 80% train-test split, the SVD model gave a RMSE of 1.6426.

-----------

We can see that using the SVD algorithm has already far out-performed the memory-based collaborative filtering approach (RMSE of 1.64 vs 7.92)!
<!-- #endregion -->

<!-- #region id="YV8IIZlAoGU5" -->
### NMF model
<!-- #endregion -->

```python id="s2zWKlNeoGU6" outputId="66a765de-68b2-4b9b-916a-40e2cd9205db"
# Load NMF algorithm
model = NMF()
# Train on books dataset
%time model_selection.cross_validate(model, data, measures=['RMSE'], cv=5, verbose=True)
```

<!-- #region id="osXflz-PoGU6" -->
The NMF model gave a mean RMSE of ca. 2.47, with a fit time of ca. 48 s.

It seems like the SVD algorithm is the best choice for this dataset.
<!-- #endregion -->

<!-- #region id="aBqAGsw2oGU7" -->
## Optimizing the SVD algorithm with parameter tuning
Since it seems like the SVD algorithm is our best choice, let's see if we can improve the predictions even further by optimizing some of the algorithm hyperparameters.

One way of doing this is to use the handy ```GridSearchCV``` method from the ```surprise``` library. When passed a range of hyperparameter values, ```GridSearchCV``` will automatically search through the parameter-space to find the best-performing set of hyperparameters.
<!-- #endregion -->

```python id="DFzfw1v4oGU8"
# We'll remake the training set, keeping 20% for testing
trainset, testset = model_selection.train_test_split(data, test_size=0.2)
```

```python id="nlGHPAD_oGU8"
### Fine-tune Surprise SVD model useing GridSearchCV
from surprise.model_selection import GridSearchCV

param_grid = {'n_factors': [80, 100, 120], 'lr_all': [0.001, 0.005, 0.01], 'reg_all': [0.01, 0.02, 0.04]}

# Optimize SVD algorithm for both root mean squared error ('rmse') and mean average error ('mae')
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
```

```python id="YVzdFkyeoGU8" outputId="079ed22e-6101-4fb3-b68a-93fa0a61bd80"
# Fit the gridsearch result on the entire dataset
%time gs.fit(data)
```

```python id="bCdg3a4xoGU9" outputId="18e23b10-82ae-42c9-887c-05682cf6c8e3"
# Return the best version of the SVD algorithm
model = gs.best_estimator['rmse']

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
```

```python id="Av7-S9PLoGU9" outputId="5e5ba574-0a86-4f77-8fbe-9b0698ffe614"
model_selection.cross_validate(model, data, measures=['rmse', 'mae'], cv=5, verbose=True)
```

<!-- #region id="138d3BTqoGU9" -->
The mean RSME using the optimized parameters was 1.6351 over 5 folds, with an average fit time of ca. 24s.
<!-- #endregion -->

```python id="f61GP2efoGU-" outputId="271001ed-bfc9-48a4-da99-adc24107f065"
### Use the new parameters with the training set
model = SVD(n_factors=80, lr_all=0.005, reg_all=0.04)
model.fit(trainset) # re-fit on only the training data using the best hyperparameters
test_pred = model.test(testset)
print("SVD : Test Set")
accuracy.rmse(test_pred, verbose=True)
```

<!-- #region id="OrzeRWVfoGU-" -->
Using the optimized hyperparameters we see a slight improvement in the resulting RMSE (1.629) compared with the unoptimized SVD algorithm (1.635)1
<!-- #endregion -->

<!-- #region id="-b0iuwt_oGU-" -->
## Testing some of the outputs (ratings and recommendations)
Would like to do an intuitive check of some of the recommendations being made.

Let's just choose a random user/book pair (represented in the ```suprise``` library as ```uid``` and ```iid```, respectively).

**Note:** The ```model``` being used here is the optimized SVD algorithm that has been fit on the training set.
<!-- #endregion -->

```python id="K0M_ohLWoGU-" outputId="d3bb1d03-8f34-4214-d6f0-c8c151ba9bd2"
# get a prediction for specific users and items.
uid = 276744  # the USERID int
iid = '038550120X' # the UNIQUE_ITEMIDS string
# This pair has an actual rating of 7!

pred = model.predict(uid, iid, verbose=True)
```

<!-- #region id="9rAoE5K2oGU_" -->
Can access the attributes of the ```predict``` method to get a nicer output.
<!-- #endregion -->

```python id="ODmTdaauoGU_" outputId="e5c3a816-02dc-4072-d34c-dadc05440b31"
print(f'The estimated rating for the book with the "UNIQUE_ITEMIDS" code {pred.iid} from user #{pred.uid} is {pred.est:.2f}.\n')
actual_rtg = user_item_rating[(user_item_rating.USERID==pred.uid) & (user_item_rating.UNIQUE_ITEMIDS==pred.iid)].RATING.values[0]
print(f'The real rating given for this was {actual_rtg:.2f}.')
```

```python id="-zv72k-1oGU_" outputId="32639170-1f4a-42bb-a4d4-980f1f04bf2b"
# get a prediction for specific users and items.
uid = 95095  # the USERID int
iid = '0140079963' # the UNIQUE_ITEMIDS string
# This pair has an actual rating of 6.0!

pred = model.predict(uid, iid, verbose=True)
```

```python id="ktgz5FCvoGVA" outputId="b77109c6-6f8e-46bd-e16e-6269b0358416"
print(f'The estimated rating for the book with the "UNIQUE_ITEMIDS" code {pred.iid} from user #{pred.uid} is {pred.est:.2f}.\n')
actual_rtg = user_item_rating[(user_item_rating.USERID==pred.uid) & (user_item_rating.UNIQUE_ITEMIDS==pred.iid)].RATING.values[0]
print(f'The real rating given for this was {actual_rtg:.2f}.')
```

<!-- #region id="AJ92N_PsoGVA" -->
The following function was adapted from the ```surprise``` docs, and can be used to get the top book recommendations for each user.
<!-- #endregion -->

```python id="f8MSn3EzoGVA"
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

<!-- #region id="zXWgKl-PoGVA" -->
Let's get the Top 10 recommended books for each USERID in the test set.
<!-- #endregion -->

```python id="KBRbKh8OoGVB"
pred = model.test(testset)
top_n = get_top_n(pred)
```

```python id="baeVCqdboGVB"
def get_reading_list(userid):
    """
    Retrieve full book titles from full 'books_users_ratings' dataframe
    """
    reading_list = defaultdict(list)
    top_n = get_top_n(predictions, n=10)
    for n in top_n[userid]:
        book, rating = n
        title = books_users_ratings.loc[books_users_ratings.UNIQUE_ITEMIDS==book].TITLE.unique()[0]
        reading_list[title] = rating
    return reading_list
```

```python id="JzGUwRqYoGVB" outputId="5b72668e-73ef-4511-9923-701abcf86e24"
# Just take a random look at USERID=60337
example_reading_list = get_reading_list(userid=60337)
for book, rating in example_reading_list.items():
    print(f'{book}: {rating}')
```

<!-- #region id="fSlDoIXnoGVC" -->
Have tried out a few different ```userid``` entries (from the ```testset```) to see what the top 10 books that user would like are and they seem pretty well related, indicating that the recommendation engine is performing reasonably well!
<!-- #endregion -->

<!-- #region id="jXopErr-oGVD" -->
# Summary
<!-- #endregion -->

<!-- #region id="VhLmXiMFoGVD" -->
In this notebook a dataset from the 'Book-Crossing' website was used to create a recommendation system. A few different approaches were investigated, including memory-based correlations, and model-based matrix factorization algorithms[2]. Of these, the latter - and particularly the Singular Value Decomposition (SVD) algorithm - gave the best performance as assessed by comparing the predicted book ratings for a given user with the actual rating in a test set that the model was not trained on.

The only fields that were used for the model were the "user ID", "book ID", and "rating". There were others available in the dataset, such as "age", "location", "publisher", "year published", etc, however for these types of recommendation systems it has often been found that additional data fields do not increase the accuracy of the models significantly[1]. A "Grid Search Cross Validation" method was used to optimize some of the hyperparameters for the model, resulting in a slight improvement in model performance from the default values.

Finally, we were able to build a recommender that could predict the 10 most likely book titles to be rated highly by a given user.

It should be noted that this approach still suffers from the "cold start problem"[3] - that is, for users with no ratings or history the model will not make accurate predictions. One way we could tackle this problem may be to initially start with popularity-based recommendations, before building up enough user history to implement the model. Another piece of data that was not utilised in the current investigation was the "implicit" ratings - denoted as those with a rating of "0" in the dataset. Although more information about these implicit ratings (for example, does it represent a positive or negative interaction), these might be useful for supplementing the "explicit" ratings recommender.
<!-- #endregion -->

<!-- #region id="OzoDlqU8oGVE" -->
# References
<!-- #endregion -->

<!-- #region id="bqfJ6uezoGVE" -->
1. http://blog.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/
2. https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html
3. https://towardsdatascience.com/building-a-recommendation-system-for-fragrance-5b00de3829da
<!-- #endregion -->
