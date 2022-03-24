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

In the [last notebook](https://github.com/caiomiyashiro/RecommenderSystemsNotebooks/blob/master/Month%202%20Part%20I%20-%20User%20User%20Collaborative%20Filtering.ipynb), we took a look at the Nearest Neighboor User User CF, a form of recommender system that looked at the similarity between users to define which item should be suggested to a new customer.

We saw the benefits from User User CF when comparing it to non personalised and content based recommendations, but we also saw that it comes with one difficulty.

- It doesn't scale well. Even in a big e-commerce dataset, the amount of intersecting items between 2 users is not as big as it could be when User User CF was created. Because of this, when a user bought an item that intersected now with another different customer, the new similarity could drastically change, causing the system's owners to recalculate the new similarities very often.
  
- If they don't update the matrix often, they can also lose profits over it because it wouldn't map the user's short term interests which, on the internet, can vary quite a lot.

All in all, short term interest and sparse mutual interest space make the User User CF inapt for high scale companies.

# Item Item Collaborative Filtering (CF)

[Item Item CF](https://en.wikipedia.org/wiki/Item-item_collaborative_filtering) was created by ([Sarwar et all, 1998](https://patentimages.storage.googleapis.com/41/80/fb/07d4d9e61e7431/US6266649.pdf)) in partneship with Amazon in order to fix the problems with the User User CF. In Item Item perspective, as the name suggests, changes the perspective from User centered to a Item centered view, *i.e.*, instead of having a User User similarity matrix, they started to use a item item similarity matrix. Then, when a user $u$ bought and liked an item $i_{1}$ and $i_{1}$ was similar to item $i_{2}$, then we predicted that $u$ would also like $i_{2}$. Take a look at the image below:

<img src="images/notebook5_image1.jpeg" width="500">

Why this simple change in perspective helped to solve the inneficiency problems present in the User User CF?

By considering an enviroment of a big e-commerce company, we end with the number of users >> number of items.   
In this case, even if a single user hasn't given many reviews, the chances are that many users have given a review to a specific item.
By having a big number of reviews, an item relationship to other items doesn't change too much by receiving a few more reviews, *i.e.*, item item relationship are more stable. Therefore, by being more stable, the similarity matrix doesn't have to be recalculated often, as in the User User CF.
  
An extra perfomance improvement comes also from the prediction calculation. In the Item Item CF, a new prediction for a user $u$ for a product $p$ is made by retrieving the items similarities and calculating a weighted average. The number of neighboors for this calculation is only the item that $u$ has liked or bought in the past and this number is often small enought. Therefore, we don't need to search the big user user similarity matrix to find the best $k$ neighboors.

# Item Item Steps

As always, we're going to work with one of the datasets from the [Coursera's Specialization on Recommender Systems](https://www.coursera.org/specializations/recommender-systems). This dataset is from the last week in the course of [Nearest Neighboors CF](https://www.coursera.org/learn/collaborative-filtering) for Item Item CF. Dataset is [here](https://d396qusza40orc.cloudfront.net/umntestsite/on-demand_files/A5/Assignment%205.xls) (Coursera's page) and [here](https://github.com/caiomiyashiro/RecommenderSystemsNotebooks/blob/master/data/Item%20Item%20Collaborative%20Filtering%20-%20Ratings.csv) (personal Github account).
  
The steps taken to evaluate and recommend are similar to User User CF, with some different calculations in the prediction step, as we've said.

- Load traditional input - User Item Review dataset
- Create similarity matrix
- Make predictions

Lets go!

## Example Dataset

The dataset is a matrix with size 25 users x 25 movies and each cell $c_{u,m}$ contains the rating user $u$ gave to movie $m$. If user $u$ didn't rate movie $m$, the cell is empty. As the float values were stored with commas and consequently were being casted as strings, I had to process it a little bit to replace the commas for dots and then convert the column to floats

```python
import pandas as pd
import numpy as np
```

```python
df = pd.read_csv('data/Item Item Collaborative Filtering - Ratings.csv', index_col=0, nrows=20)

df.drop('Mean', axis=1, inplace=True) # remove mean column that comes at the end

# replace commas for dots and convert previous string column into float
def processCol(col):
    return col.astype(str).apply(lambda val: val.replace(',','.')).astype(float)
df = df.apply(processCol)

print('Dataset shape: ' + str(df.shape))
df.head()
```

<!-- #region -->
## Create Similarity Matrix


### Similarity Function

As for the User User CF, we have a few possibilities to choose from when deciding how we're going to define if an item is similar to another item. Again, ([Herlocker et all, 2002](https://grouplens.org/site-content/uploads/evaluating-TOIS-20041.pdf)) did an analysis on the performance of these metrics on Item Item CF and realised that, for this case, the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) was the best performant metric. So we're going with them this time. On the **next notebook**, we try to analyse these metrics and see why it performs better in certain cases and others not.


## Calculating User User Similarity with Cosine Similarity:

One important point here is on the calculation of the denominator of the cosine similarity. Even though we make the dot product only with values existing in both arrays, the norm of the individual vectors are considering all values, and not the intersections between array1 and array2.
<!-- #endregion -->

```python
def cos_similarity(item1, item2):
    item1Values = ~np.isnan(item1)
    item2Values = ~np.isnan(item2)
    allValues = np.logical_and(item1Values,item2Values) # get only existent elements of both vectors
    return np.dot(item1[allValues], item2[allValues])/(np.linalg.norm(item1[item1Values]) * np.linalg.norm(item2[item2Values]))

def pre_cos_similarity(item1, df):
    return df.apply(lambda item2: cos_similarity(item1, item2))

df_corr = df.apply(lambda item1: pre_cos_similarity(item1, df))
df_corr.head()
```

## Predictions Calculation

By now we already know which items are more similar to each other. This will help us when predicting a new rating, by giving higher weights for more similar items than other the user has bough. 
  
The way we're going to calculate the new predictions is the same we used for User User CF, *i.e.*, a weighted average:

$$\frac{\sum_{n=1}^{k} r_{n}w_{n}}{\sum_{n=1}^{k} w_{n}}$$
  
The difference is that we don't have the neighboors anymore, so the $n$ in the summation is considering **all** the items user $u$ has rated and $w$ is still the similarities, but now *item similarity*.

```python
def predictRating(userRatings, itemSimilarity):
    userHasRating = ~np.isnan(userRatings)
    return np.dot(userRatings[userHasRating], itemSimilarity[userHasRating])/np.sum(itemSimilarity[userHasRating])

def pre_predictRating(userRatings, df_corr):
    return df_corr.apply(lambda itemSimilarity: predictRating(userRatings, itemSimilarity))

predictions = df.apply(lambda userRatings: pre_predictRating(userRatings, df_corr), axis=1)
predictions.head()    
```

## Mean Normalised Weighted Average

As in the same way of the User User CF, we can calculate predictions using the absolute value of the reviews or from the mean centralised values of it. The advantages are the same: consider the scale variability of reviewers when attributing a final score for a item of interest:

$$\bar{r_{u}} + \frac{\sum_{n=1}^{k} (r_{n} - \bar{r_{n}})w_{n}}{\sum_{n=1}^{k} w_{n}}$$

We took the same function as above, but added two extra parameters:
- $userMeanRating$: mean average ratings for a specific user
- $neighboorsMeanRating$: mean average rating for all the nearest neighboors for a specific user

```python
# mean normalise
def subtractFromMean(col, meanCol):
    result = np.array([np.nan] * col.shape[0])
    isValidValue = ~np.isnan(col)
    result[isValidValue] = col.values[isValidValue] - meanCol.values[isValidValue]
    return result
userMeanRatings = df.apply(np.mean, axis=1)
df_ratings_norm = df.apply(lambda col: subtractFromMean(col, userMeanRatings))

# similarity matrix
df_corr_norm = df_ratings_norm.apply(lambda item1: pre_cos_similarity(item1, df_ratings_norm))

```

### Remove negative correlations

In this example, we are replacing the negative correlations by 0, as we can interpret as a maximum weight for unwanted items:


```python
def replaceNegative(col):
    col[col < 0] = 0
    return col
df_corr_norm2 = df_corr_norm.apply(replaceNegative)
```

### Predict!

```python
predictions_norm = df.apply(lambda userRatings: pre_predictRating(userRatings, df_corr_norm2), axis=1)
predictions_norm.head() 

```

* I didn't quite understand why we didn't use the not normalised item's ratings in this calculation. Ideally, we would use the mean centered user ratings as well..

<!-- #region -->
# Comparison Between Approaches

The comparison follows the same guidelines we used when evaluating the User User CF. 


# Final Considerations on User User CF

When we mean centered the user's rating for the User User CF, the objective was clear, we wanted to take into account that users rate in different parts of the scale. But what about mean centering for the Item Item CF? This evaluation I'll leave it to the next notebook, where we evaluate and compare the main metrics used for similarity calculation in a CF system.

-- 

Item Item CF brings efficiencies steps forward from the User User CF schema. With it, we bring personalised recommendations and in a way that is computationally efficient to scale for giant e-commerce companies, such as Amazon or Netflix. But Item Item CF isn't a gold system, where we can implement it and always get good results. It has a few premisses:

- First, it has the premisse that number of users >> number of items. This is a prerequisite to have stable entities, items in this case, and doesn't need to recalculate the similarity matrix often, as in the User User CF.
  
  
- Secondly, and this is an interesting feature, Item Item CF is better when the item ratings are stable, *i.e.*, they have lots of evaluations. This means that the user's items are probably going to have a lot of influence from these popular items and, at the end, receiving popular items recommendations. This is good when you want to be safe about your recommendations, such as expensive services or products or rarely bought, such as houses or cars. However, this lack of '*serendipity*' is missed when we want to enable users to find that particular rare item and amazingly matched with your tastes. As an example, If we take Spotify, we don't want to receive recommendations such as 'Hey, as you listened to Mozart, here is what we think you'd like: Bach'. Spotify greatness works on the premisse of finding the bands and songs that can surprise you, so they wouldn't be effective by working on the Item Item CF schema. Of course, we are going to see more advance techniques in the future where these companies apply modern algorithms to have good recommendations and still be performatic, but the idea now was to show how we can't rely on one algorithm as the best of them all.

<img src="images/notebook5_image2.png" width="500">
  
  
In the [next notebook](https://github.com/caiomiyashiro/RecommenderSystemsNotebooks/blob/master/Month%202%20Part%20III%20-%20Notes%20on%20Similarity%20Metrics%20for%20CF.ipynb), we finalise the discussion over Collaborative Filtering by investigating a little more on how the similarity metrics work and try to find out some of its features such as:  
  
* Why pearson end up being better for User User CF and Cosine Similarity better for Item Item CF?
* What are the strenghts and weakness when thinking on using one of the evaluated metrics?
* Some filosophies on what they represent and how we can think about them geometrically
  
Stay tuned :)
<!-- #endregion -->
