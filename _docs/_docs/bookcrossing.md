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

<!-- #region id="v8HwBi6xf4em" -->
# Book Recommender
<!-- #endregion -->

```python id="BEqUeUTDUYlW"
import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split
```

```python id="C6IGddcvUYlr"
user = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
user.columns = ['userID', 'Location', 'Age']
rating = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
rating.columns = ['userID', 'ISBN', 'bookRating']
```

```python id="cQB_Kb2_UYly" outputId="39b534a8-1294-48ff-d96d-34c7b650a36a"
user.head()
```

```python id="HSF-E0_YUYl6" outputId="947cc4d3-68b5-449e-8cd2-5f54628652c4"
rating.head()
```

```python id="z9Zh7GQ1UYmA"
df = pd.merge(user, rating, on='userID', how='inner')
df.drop(['Location', 'Age'], axis=1, inplace=True)
```

```python id="AzXS_36-UYmG" outputId="de916898-364a-49ab-fa58-5cecf26b5d10"
df.head()
```

```python id="8jpXth2OUYmM" outputId="331c95f4-69a8-4a12-a8a7-e6596350097c"
df.shape
```

```python id="q67vJ0K3UYmT" outputId="9bf30d89-040a-4e83-87b3-3784dce09a6c"
df.info()
```

```python id="P4kpmfH_UYmY" outputId="9d4da999-a678-421c-9377-6caed49627a6"
print('Dataset shape: {}'.format(df.shape))
print('-Dataset examples-')
print(df.iloc[::200000, :])
```

<!-- #region id="LmW3okF8UYme" -->
## EDA

### Ratings Distribution
<!-- #endregion -->

```python id="4-eWLH-WUYmf" outputId="79fb0b21-6967-4b8b-b7be-96d1f5f76c2b"
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

data = df['bookRating'].value_counts().sort_index(ascending=False)
trace = go.Bar(x = data.index,
               text = ['{:.1f} %'.format(val) for val in (data.values / df.shape[0] * 100)],
               textposition = 'auto',
               textfont = dict(color = '#000000'),
               y = data.values,
               )
# Create layout
layout = dict(title = 'Distribution Of {} book-ratings'.format(df.shape[0]),
              xaxis = dict(title = 'Rating'),
              yaxis = dict(title = 'Count'))
# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
```

<!-- #region id="NtdP-FlQUYml" -->
We can see that over 62% of all ratings in the data are 0, and very few ratings are 1 or 2, or 3, low rating books mean they are generally really bad.
<!-- #endregion -->

<!-- #region id="gN6yoHYJUYmm" -->
### Ratings Distribution By Book
<!-- #endregion -->

```python id="J3Iifx0iUYmn" outputId="3b6a1e24-92ec-4175-95bf-06749563eb79"
# Number of ratings per book
data = df.groupby('ISBN')['bookRating'].count().clip(upper=50)

# Create trace
trace = go.Histogram(x = data.values,
                     name = 'Ratings',
                     xbins = dict(start = 0,
                                  end = 50,
                                  size = 2))
# Create layout
layout = go.Layout(title = 'Distribution Of Number of Ratings Per Book (Clipped at 50)',
                   xaxis = dict(title = 'Number of Ratings Per Book'),
                   yaxis = dict(title = 'Count'),
                   bargap = 0.2)

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
```

```python id="PpiLsJeZUYmu" outputId="8799a51e-c1bb-4dcb-bacd-993d56a88e2e"
df.groupby('ISBN')['bookRating'].count().reset_index().sort_values('bookRating', ascending=False)[:10]
```

<!-- #region id="-NheBcYPUYmz" -->
Most of the books received less than 5 ratings, and very few books have many ratings, although the most rated book has received 2,502 ratings.
<!-- #endregion -->

<!-- #region id="S7eisCwEUYm0" -->
### Ratings Distribution By User
<!-- #endregion -->

```python id="Bz81hZlRUYm1" outputId="140d5b61-9a0e-428b-b9b3-10212b825498"
# Number of ratings per user
data = df.groupby('userID')['bookRating'].count().clip(upper=50)

# Create trace
trace = go.Histogram(x = data.values,
                     name = 'Ratings',
                     xbins = dict(start = 0,
                                  end = 50,
                                  size = 2))
# Create layout
layout = go.Layout(title = 'Distribution Of Number of Ratings Per User (Clipped at 50)',
                   xaxis = dict(title = 'Ratings Per User'),
                   yaxis = dict(title = 'Count'),
                   bargap = 0.2)

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
```

```python id="Ne_8BQtKUYm6" outputId="68cf83bf-de30-413b-899a-35c1df86effc"
df.groupby('userID')['bookRating'].count().reset_index().sort_values('bookRating', ascending=False)[:10]
```

<!-- #region id="IMv7Y7uSUYm_" -->
Most of the users gave less than 5 ratings, and very few users gave many ratings, although the most productive user have given 13,602 ratings.
<!-- #endregion -->

<!-- #region id="QdNthqK5UYnA" -->
I'm sure you have noticed that the above two charts share the same distribution. The number of ratings per movie and the number of ratings per user decay exponentially.
<!-- #endregion -->

<!-- #region id="6Eg7wxBvUYnB" -->
To reduce the dimensionality of the dataset, we will filter out rarely rated movies and rarely rating users.
<!-- #endregion -->

```python id="AwixGILkUYnC" outputId="7e813c0c-e921-4407-c60a-2609305cb55e"
min_book_ratings = 50
filter_books = df['ISBN'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

min_user_ratings = 50
filter_users = df['userID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

df_new = df[(df['ISBN'].isin(filter_books)) & (df['userID'].isin(filter_users))]
print('The original data frame shape:\t{}'.format(df.shape))
print('The new data frame shape:\t{}'.format(df_new.shape))
```

<!-- #region id="wsm95t8LUYnI" -->
## Surprise

To load a dataset from a pandas dataframe, we will use the load_from_df() method, we will also need a Reader object, and the rating_scale parameter must be specified. The dataframe must have three columns, corresponding to the user ids, the item ids, and the ratings in this order. Each row thus corresponds to a given rating.
<!-- #endregion -->

```python id="3oR07ORpUYnJ"
reader = Reader(rating_scale=(0, 9))
data = Dataset.load_from_df(df_new[['userID', 'ISBN', 'bookRating']], reader)
```

<!-- #region id="_gMo8XYuUYnO" -->
With the Surprise library, we will benchmark the following algorithms

### Basic algorithms

#### NormalPredictor

* NormalPredictor algorithm predicts a random rating based on the distribution of the training set, which is assumed to be normal. This is one of the most basic algorithms that do not do much work.

#### BaselineOnly

* BasiclineOnly algorithm predicts the baseline estimate for given user and item.

### k-NN algorithms

#### KNNBasic

* KNNBasic is a basic collaborative filtering algorithm.

#### KNNWithMeans

* KNNWithMeans is basic collaborative filtering algorithm, taking into account the mean ratings of each user.

#### KNNWithZScore

* KNNWithZScore is a basic collaborative filtering algorithm, taking into account the z-score normalization of each user.

#### KNNBaseline

* KNNBaseline is a basic collaborative filtering algorithm taking into account a baseline rating.

### Matrix Factorization-based algorithms

#### SVD

* SVD algorithm is equivalent to Probabilistic Matrix Factorization (http://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf)

#### SVDpp

* The SVDpp algorithm is an extension of SVD that takes into account implicit ratings.

#### NMF

* NMF is a collaborative filtering algorithm based on Non-negative Matrix Factorization. It is very similar with SVD.

### Slope One

* Slope One is a straightforward implementation of the SlopeOne algorithm. (https://arxiv.org/abs/cs/0702144)

### Co-clustering

* Co-clustering is a collaborative filtering algorithm based on co-clustering (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.113.6458&rep=rep1&type=pdf)


We use rmse as our accuracy metric for the predictions.
<!-- #endregion -->

```python id="tZ9R9XDyUYnP" outputId="543f1940-6183-4007-826c-b18570d8e050"
benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
```

```python id="8Xo2BkItUYnU"
surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
```

```python id="sDRCQt5kUYnY" outputId="10010efa-6059-4098-8542-93a387125ab7"
surprise_results
```

<!-- #region id="Zpolrs3BUYnd" -->
BaselineOnly algorithm gave us the best rmse, therefore, we will proceed further with BaselineOnly and use Alternating Least Squares (ALS).
<!-- #endregion -->

```python id="cBzfW9XEUYne" outputId="a9a06dda-c451-4694-d4f3-140464a9ac53"
print('Using ALS')
bsl_options = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }
algo = BaselineOnly(bsl_options=bsl_options)
cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)
```

<!-- #region id="DSKogVzOUYnj" -->
We use the train_test_split() to sample a trainset and a testset with given sizes, and use the accuracy metric of rmse. We’ll then use the fit() method which will train the algorithm on the trainset, and the test() method which will return the predictions made from the testset
<!-- #endregion -->

```python id="bk3e7qU_UYnk" outputId="c042fbc5-743b-4c00-e64a-8fe96aa83ffb"
trainset, testset = train_test_split(data, test_size=0.25)
algo = BaselineOnly(bsl_options=bsl_options)
predictions = algo.fit(trainset).test(testset)
accuracy.rmse(predictions)
```

```python id="aivyoO0UUYnp"
# dump.dump('./dump_file', predictions, algo)
# predictions, algo = dump.load('./dump_file')
```

```python id="li2SgaK_UYnt" outputId="a373aeb1-b298-40c7-f13d-bee16ea2ea27"
trainset = algo.trainset
print(algo.__class__.__name__)
```

<!-- #region id="XG4FZpSJUYn0" -->
To inspect our predictions in details, we are going to build a pandas data frame with all the predictions.
<!-- #endregion -->

```python id="xdnIsGTFUYn1"
def get_Iu(uid):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
    
df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)
```

```python id="yzC7pNCrUYn6" outputId="788d06f8-43e3-42dc-cdc0-1719e4a470f6"
df.head()
```

```python id="_IIAH6rzUYn-"
best_predictions = df.sort_values(by='err')[:10]
worst_predictions = df.sort_values(by='err')[-10:]
```

```python id="on8of035UYoC" outputId="8cd38905-cc8d-4215-e165-f0d0ce4ae2a6"
best_predictions
```

<!-- #region id="fn6udPQPUYoH" -->
The above are the best predictions, and they are not lucky guesses. Because Ui is anywhere between 26 to 146, they are not really small, meaning that significant number of users have rated the target book.
<!-- #endregion -->

```python id="l4adM94nUYoJ" outputId="d16613f1-dd96-4a3f-f18f-464b58c1116e"
worst_predictions
```

<!-- #region id="f-5GAn7_UYoN" -->
The worst predictions look pretty surprise. Let's look in more details of the last one ISBN "055358264X", the book was rated by 47 users, user "26544" rated 10, our BaselineOnly algorithm predicts 0.
<!-- #endregion -->

```python id="GVOIW08OUYoO" outputId="450ef8f9-69c6-444d-8f39-8dc0512db9fd"
df_new.loc[df_new['ISBN'] == '055358264X']['bookRating'].describe()
```

```python id="8ohLvVqnUYoS" outputId="217572fe-9500-40e5-b26e-d1cd06de7b1e"
import matplotlib.pyplot as plt
%matplotlib notebook

df_new.loc[df_new['ISBN'] == '055358264X']['bookRating'].hist()
plt.xlabel('rating')
plt.ylabel('Number of ratings')
plt.title('Number of ratings book ISBN 055358264X has received')
plt.show();
```

<!-- #region id="kAYe7M1zUYoX" -->
It turns out, most of the ratings this book received was "0", in another word, most of the users in the data rated this book "0", only very few users rated "10". Same with the other predictions in "worst predictions" list. It seems that for each prediction, the users are some kind of outsiders.
<!-- #endregion -->
